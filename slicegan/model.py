from slicegan import util
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim as optim
import time
import matplotlib
matplotlib.use('Agg')
import json
import itertools

#wandb.login() must be performed on the machine to store login information under .netrc
import wandb, netrc
# import pytorch_lightning as pl
# import torchmetrics
# pl.seed_everything(hash("setting random seeds") % 2**32 - 1)
# from pytorch_lightning.loggers import WandbLogger

wandb_support = False
wandb_username = "josh_stickel"
wandb_projectname = "SliceGAN"

def initialize_wandb(data):

    # check that user has logged in
    if not "api.wandb.ai" in netrc.netrc().hosts:
        raise RuntimeError("No wandb entry in \".netrc\". User has not logged into wandb on this machine before.")

    # initialize and specify hyperparameters to log
    wandb.init(project=wandb_projectname, entity=wandb_username)
    wandb.config = {
        "batch_size": data["batch_size"],
        "lrg"       : data["lrg"       ],
        "lrd"       : data["lrd"       ],
        "G_beta_1"  : data["G_beta_1"  ],
        "G_beta_2"  : data["G_beta_2"  ],
        "D_beta_1"  : data["D_beta_1"  ],
        "D_beta_2"  : data["D_beta_2"  ],
        "Lambda"    : data["Lambda"    ]
    }

def train(path_input, pth, imtype, datasets, Disc, Gen, nc, l, nz, n_dims, Normalize=True, split_graphs=False):

    """
    train the generator
    :param pth: path to save all files, imgs and data
    :param imtype: image type e.g nphase, colour or gray
    :param datatype: training data format e.g. tif, jpg ect
    :param real_data: path to training data
    :param Disc:
    :param Gen:
    :param nc: channels
    :param l: image size
    :param nz: latent vector size
    :param sf: scale factor for training data
    :return:
    """
    
    # Import user variables
    with open(path_input, 'r') as f:
        data = json.load(f)["model"]
    # number of training loops to go through
    num_epochs   = data["num_epochs"  ]
    # number of times the critic should be updated before the generator is updated
    # in normal GANs, the generator would be updated more, but WGANs benefit from
    # more powerful critics
    critic_iters = data["critic_iters"]
    # number of images to keep in memory at once
    batch_size   = data["batch_size"  ]
    # Optimizer learning rate
    lrg          = data["lrg"         ]
    lrd          = data["lrd"         ]
    # Optimizer betas
    G_beta_1     = data["G_beta_1"    ]
    G_beta_2     = data["G_beta_2"    ]
    D_beta_1     = data["D_beta_1"    ]
    D_beta_2     = data["D_beta_2"    ]
    # learning parameter for the gradient penalty
    Lambda       = data["Lambda"      ]
    ### Parallelization
    # number of gpus to use (if available)
    ngpu         = data["ngpu"        ]
    # data loader workers
    workers      = data["workers"     ]
    
    G_betas = [G_beta_1, G_beta_2]
    D_betas = [D_beta_1, D_beta_2]
    
    # Wandb Integration
    if wandb_support:
        initialize_wandb(data)
    
    # Fill in plane information for 2D or 3D
    if n_dims == 2:
        n_planes = 1
        plane_names = ['Z']
        c_perm = [
            [0, 1, 2, 3] # don't permute
        ]
        # The discriminator batch size is a portion of the generator batch size following: n*2^(n_planes-1)=batch_size ==> n
        D_batch_size = int(batch_size/(2**(n_planes-1)))
        shape_disc = [D_batch_size, nc, l, l]
        shape_gen  = [batch_size  , nc, l, l]
    elif n_dims == 3:
        n_planes = 3
        plane_names = ['X','Y','Z']
        # permutation constants to convert 3D volume into a batch of 2D images
        # data is of shape: [batch_size, nc, z, y, x]
        c_perm = [
            [0, 2, 1, 3, 4], # last 3 indecies are now [nc,y,x] -> z normal 
            [0, 3, 1, 2, 4], # last 3 indecies are now [nc,z,x] -> y normal
            [0, 4, 1, 2, 3]  # last 3 indecies are now [nc,z,y] -> x normal
        ]
        # The discriminator batch size is a portion of the generator batch size following: n*2^(n_planes-1)=batch_size ==> n
        D_batch_size = int(batch_size/(2**(n_planes-1)))
        shape_disc = [l * D_batch_size, nc, l, l] # combine batch size and slice normal plane (turn the entire unused direction into batches)
        shape_gen  = [l * batch_size  , nc, l, l]

    ########## change to something with "torch.cuda.device_count()" ##########
    ## Switch to cuda if available
    device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")
    if(torch.cuda.device_count() > 0 and torch.cuda.is_available()):
        print(torch.cuda.device_count(), " ", device, " devices will be used")
    else:
        print(device, " will be used.")

    ## Dataloaders for each orientation
    # D trained using different data for x, y and z directions
    dataloader = []
    for i in range(n_planes):
        dataloader.append(
            torch.utils.data.DataLoader(
                datasets[i],
                batch_size=batch_size,
                num_workers=workers
            )
        )

    # Create the Genetator network and Generator optimizer
    netG = Gen().to(device)
    print("generator:\n",netG)
    ########## change to something with "torch.cuda.device_count()" ##########
    if ('cuda' in str(device)) and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    optG = optim.Adam(netG.parameters(), lr=lrg, betas=G_betas)
    
    # Spatial dimensions of the latent space vector required to match the image size after deconvolution
    lz = util.find_min_input_size_deconvolution(netG.kernel_size,netG.stride,netG.padding,"lz",l)

    # Create the Discriminator networks and Discriminator optimizers (1 for each orthogonal plane)
    netDs = []
    optDs = []
    for i in range(n_planes):
        netD = Disc().to(device)
        print("discriminator for {0} plane:\n".format(plane_names[i]),netD)
        ########## change to something with "torch.cuda.device_count()" ##########
        if ('cuda' in str(device)) and (ngpu > 1):
            netD = (nn.DataParallel(netD, list(range(ngpu))))
        netDs.append(netD)
        optDs.append(optim.Adam(netDs[i].parameters(), lr=lrd, betas=D_betas))

    # Initialize logging variables
    disc_loss_log = []
    gp_log = []
    Wass_log = []
    for dim in range(n_planes):
        disc_loss_log.append([[],[]])
        gp_log.append([])
        Wass_log.append([])
    
    # Log start time to calculate ETA
    time_start = time.time()
    
    print("Starting Training Loop...")
    
    for epoch in range(num_epochs):
    
        # sample data for each direction
        for i, dataset in enumerate(list(zip(*dataloader)), 1):

            ## Generate a noise and fake image
            # Only one generation is needed because all 3 (orthogonal) discriminators will use it
            noise = torch.randn(D_batch_size, nz, *[lz]*n_dims, device=device)
            # Must be detached because otherwise it become associated with the first discriminator's graph
            # and cannot be called again for the 2nd and 3rd discriminator
            data_fake = netG(noise).detach()

            for dim, (netD, optD, data_real, c_perm_dim) in enumerate(zip(netDs, optDs, dataset, c_perm)):
                
                # Zero out the gradient
                netD.zero_grad()
                
                ### Forward pass
                ## train on real images
                data_real = data_real[0].to(device)
                out_real = netD(data_real).mean()
                ## perform permutation + reshape to turn volume into batch of 2D images to pass to D
                data_fake_perm = data_fake.permute(*c_perm_dim).reshape(*shape_disc)
                ## train on fake images
                out_fake = netD(data_fake_perm)
                out_fake = out_fake.mean()
                ## loss criterion for wgan (as opposed to BCEWithLogitsLoss for gan)
                gradient_penalty = util.calc_gradient_penalty(netD, data_real, data_fake_perm[:batch_size],
                                                                      batch_size, l,
                                                                      device, Lambda, nc)
                disc_cost = out_fake - out_real + gradient_penalty
                
                ### Backward pass
                disc_cost.backward()
                
                ### Step the optimizer
                optD.step()
            
                #collect losses
                disc_loss_log[dim][0] += [out_real.item()]; disc_loss_log[dim][1] += [out_fake.item()]
                Wass_log[dim] += [out_real.item()-out_fake.item()]
                gp_log[dim] += [gradient_penalty.item()]
            
            if wandb_support:
                wandb.log({
                    "Discriminator loss - Real": out_real.item(),
                    "Discriminator loss - Fake": out_fake.item(),
                    "Wasserstein Distance"     : out_real.item()-out_fake.item(),
                    "Gradient Penalty"         : gradient_penalty.item()
                    })
                
            
            
            ### Generator Training
            #in a wasserstein gan, the critic (unlike the discriminator) cannot overpower the generator
            #training the critic more more can only benefit the balance of the model. 
            #this is why it looks backwards in comparison to a normal gan.
            if i % int(critic_iters) == 0:
                
                # generate noise and fake image
                noise = torch.randn(batch_size, nz, *[lz]*n_dims, device=device)
                data_fake = netG(noise)
                
                # zero out the gradient
                netG.zero_grad()

                #forward pass
                errG = 0
                for netD, c_perm_dim in zip(netDs, c_perm):
                    # permute and reshape to feed to disc
                    data_fake_perm = data_fake.permute(*c_perm_dim).reshape(*shape_gen)
                    output = netD(data_fake_perm)
                    output = output.mean()
                    errG += output
                    
                # backward pass
                errG.backward()
                
                # step the optimizer
                optG.step()
                

            # Output training stats & show imgs
            if( i % 25 == 0 or i == num_epochs - 1):
            
                netG.eval() #turn off training mode
                with torch.no_grad():
                
                    # Save the generator and discriminator weights
                    save_weights(pth, plane_names, netG, netDs)

                    # Save graphs of loss related data
                    graphs = {
                        "LossGraph":{
                            "data":disc_loss_log,
                            "labels": ["Real Dataset Error", "Generated Dataset Error"]
                            },
                        "WassGraph":{
                            "data":Wass_log,
                            "labels": ["Wasserstein Distance"]
                            },
                        "GpGraph":{
                            "data":gp_log,
                            "labels":["Gradient Penalty"]
                            }
                        }
                    save_graphs(pth, plane_names, graphs, split_graphs)
                    
                    # Save image of current slices (if the number of channels allows for it)
                    if nc <= 4:
                        noise = torch.randn(1, nz, *[lz]*n_dims, device=device)
                        img = netG(noise)
                        util.test_plotter(img, imtype, pth, n_dims=n_dims, slcs=5)
                    
                    # Print the estimated time remaining
                    util.calc_eta(len(dataloader[0]), time.time(), time_start, i, epoch, num_epochs)

                netG.train() #turn on training mode

def batch_train_discriminator(netD, optDs, data_real, data_fake, perm_dim, shape_disc, device):
    pass

def save_graphs(pth, plane_names, graphs, split_graphs=False):
    
        for graph_title, graph_data in zip(graphs.keys(), graphs.values()):
        
            # plot each plane in its own graph
            if split_graphs:
                
                for plane_name, plane_data in zip(plane_names, graph_data["data"]):
                    
                    # graph_plot must be sent a list of lists
                    if not type(plane_data[0]) in [list,tuple]:
                        plane_data = [plane_data]
                
                    util.graph_plot(plane_data, graph_data["labels"], pth, graph_title+'_'+plane_name)
                    
            # plot all planes on the same graph
            else:
            
                # elevate the plane sublists
                # ex:
                #     input =  [ 
                #                 [ # x plane
                #                    [var1],[var2]
                #                 ],
                #                 [ # y plane
                #                    [var3],[var4]
                #                 ],
                #                 [ # z plane
                #                    [var5],[var6]
                #                 ]
                #              ]
                #     output = [[var1],[var2],[var3],[var4],[var5],[var6]]
                if type(graph_data["data"][0][0]) in [list, tuple]:
                    graph_data["data"] =  sum(graph_data["data"],[])
                    
                # permute the labels and planes
                # ex: 
                #     input : labels = ["real", "generated"], planes = ["x","y","z"]
                #     output: labels = ["x_real", "x_generated", "y_real", "y_generated", "z_real", "z_generated"]
                plane_labels = [plane_name+" Normal" for plane_name in plane_names]
                labels = util.permute(plane_labels, graph_data["labels"])
                    
                util.graph_plot(graph_data["data"], labels, pth, graph_title)
    
def save_weights(pth, plane_names, netG, netDs):

    #save Generator model weights
    torch.save(netG.state_dict(), pth + '_Gen.pt')
    
    #save Discriminator model weights
    for plane_name, netD in zip(plane_names, netDs):
        torch.save(netD.state_dict(), pth + '_Disc_'+plane_name+'.pt')
