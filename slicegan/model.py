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
    
    if n_dims == 2:
        n_planes = 1
        plane_names = ['Z']
    elif n_dims == 3:
        n_planes = 3
        plane_names = ['X','Y','Z']
        # permutation constants to convert 3D volume into a batch of 2D images
        c_perm = [
            [0, 2, 1, 3, 4],
            [0, 3, 1, 2, 4],
            [0, 4, 1, 2, 3]
        ]
        ########## should write something to specify isotropy dimensions ##########
        # if len(datasets) == 1:
            # datasets *= 3
            # isotropic = True
        # else:
            # isotropic = False
            
    # The discriminator batch size is a portion of the generator batch size following: n*2^(n_planes-1)=batch_size ==> n
    D_batch_size = int(batch_size/(2**(n_planes-1)))
    
    ########## change to something with "torch.cuda.device_count()" ##########
    ##Switch to cuda if available
    device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")
    if(torch.cuda.device_count() > 0 and torch.cuda.is_available()):
        print(torch.cuda.device_count(), " ", device, " devices will be used")
    else:
        print(device, " will be used.")

    ##Dataloaders for each orientation
    # D trained using different data for x, y and z directions
    dataloader = []
    for i in range(n_planes):
        dataloader.append(
            torch.utils.data.DataLoader(datasets[i], batch_size=batch_size,
                                        shuffle=True, num_workers=workers)
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
        

    print("Starting Training Loop...")
    
    disc_loss_log = []
    gp_log = []
    Wass_log = []
    for dim in range(n_planes):
        disc_loss_log.append([[],[]])
        gp_log.append([])
        Wass_log.append([])
    
    # For each epoch
    time_start = time.time()
    for epoch in range(num_epochs):
    
        # sample data for each direction
        for i, dataset in enumerate(list(zip(*dataloader)), 1):

            ### Initialize Discriminator
            ## Generate a noise vector
            noise = torch.randn(D_batch_size, nz, *[lz]*n_dims, device=device)
            ## Generate fake image batch with G
            # Only one generation is needed because all 3 (orthogonal) discriminators will use it
            fake_data = netG(noise).detach()
            print("fake data shape: ", fake_data.shape)
            
            for dim, (netD, optD, data) in enumerate(zip(netDs, optDs, dataset)):
                
                # ########## should write something to specify isotropy dimensions ##########
                # ########## there will be tripling of discriminator iterations in this implementation ##########
                # if isotropic:
                    # netD = netDs[0]
                    # optD = optDs[0]
                
                #zero out the gradient                
                netD.zero_grad()
                
                ###forward pass
                ## train on real images
                real_data = data[0].to(device)
                out_real = netD(real_data).view(-1).mean()
                ## train on fake images
                if n_dims == 2:
                    fake_data_perm = fake_data
                if n_dims == 3:
                    # perform permutation + reshape to turn volume into batch of 2D images to pass to D
                    fake_data_perm = fake_data.permute(*c_perm[dim]).reshape(l * D_batch_size, nc, l, l).to(device) ########## added a .to(device) here because it was previously detached ##########
                out_fake = netD(fake_data_perm).mean()
                ## loss criterion for wgan (as opposed to BCEWithLogitsLoss for gan)
                gradient_penalty = util.calc_gradient_penalty(netD, real_data, fake_data_perm[:batch_size],
                                                                      batch_size, l,
                                                                      device, Lambda, nc)
                disc_cost = out_fake - out_real + gradient_penalty
                
                ## gradient descent
                # calculate the gradient
                disc_cost.backward()
                # follow (descend into) the gradient
                optD.step()
                
                #collect losses
                disc_loss_log[dim][0] += [out_real.item()]; disc_loss_log[dim][1] += [out_fake.item()]
                Wass_log[dim] += [out_real.item()-out_fake.item()]
                gp_log[dim] += [gradient_penalty.item()]
            
            ### Generator Training
            #in a wasserstein gan, the critic (unlike the discriminator) cannot overpower the generator
            #training the critic more more can only benefit the balance of the model. 
            #this is why it looks backwards in comparison to a normal gan.
            if i % int(critic_iters) == 0:
                
                # zero out the gradient
                netG.zero_grad()
                errG = 0
                
                #forward pass
                noise = torch.randn(batch_size, nz, *[lz]*n_dims, device=device)
                fake = netG(noise)
                #
                for dim, netD in enumerate(netDs):
                        
                    # ########## should write something to specify isotropy dimensions ##########
                    # if isotropic:
                        # #only need one D
                        # netD = netDs[0]
                    
                    # permute and reshape to feed to disc
                    if n_dims == 2:
                        fake_data_perm = fake
                    elif n_dims == 3:
                        fake_data_perm = fake.permute(*c_perm[dim]).reshape(l * batch_size, nc, l, l)
                    output = netD(fake_data_perm)
                    errG -= output.mean()
                    
                # gradient descent
                errG.backward()
                optG.step()

            # Output training stats & show imgs
            if( i % 25 == 0 or i == num_epochs - 1):
                netG.eval() #turn off training mode
                with torch.no_grad():
                
                    #save model weights
                    torch.save(netG.state_dict(), pth + '_Gen.pt')
                    for dim, netD in enumerate(netDs):
                        torch.save(netD.state_dict(), pth + '_Disc_'+plane_names[dim]+'.pt')

                    ###Print progress
                    ## calc ETA
                    steps = len(dataloader[0])
                    util.calc_eta(steps, time.time(), time_start, i, epoch, num_epochs)
                    
                    ###save example slices
                    noise = torch.randn(1, nz, *[lz]*n_dims, device=device)
                    img = netG(noise)
                    if n_dims == 2:
                        pass
                        ####################################################
                        ########## optionally new test plots here ##########
                        ####################################################
                    if n_dims == 3:
                        util.test_plotter(img, 5, imtype, pth)
                    
                    # plotting graphs
                    if split_graphs:
                        for disc_loss, Wass, gp, plane_name in zip(disc_loss_log, Wass_log, gp_log, plane_names):
                            util.graph_plot(disc_loss, ['real', 'perp'],     pth, 'LossGraph'+'_'+plane_name)
                            util.graph_plot([Wass],    ['Wass Distance'],    pth, 'WassGraph'+'_'+plane_name)
                            util.graph_plot([gp],      ['Gradient Penalty'], pth, 'GpGraph'  +'_'+plane_name)
                    else:
                        util.graph_plot(sum(disc_loss_log,[]), util.permute(plane_names, ['real', 'perp']    ), pth, 'LossGraph')
                        util.graph_plot(Wass_log,              util.permute(plane_names, ['Wass Distance']   ), pth, 'WassGraph')
                        util.graph_plot(gp_log,                util.permute(plane_names, ['Gradient Penalty']), pth, 'GpGraph')
                                        
                netG.train() #turn on training mode
