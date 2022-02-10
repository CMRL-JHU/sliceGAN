from slicegan import util
import torch.nn as nn
import numpy as np
import json

def slicegan_nets(path_input, imtype, img_size, img_channels, z_channels, n_dims):
    """
    Define a generator and Discriminator
    :param Training: If training, we save params, if not, we load params from previous.
    This keeps the parameters consistent for older models
    :return:
    """
    
    with open(path_input, 'r') as f:
        data = json.load(f)["networks"]
    # number of layers in G and D (generator and discriminator)
    # must remain 5 due to transpose convolution as outlined in paper below:
    # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    lays = data["lays"]
    # kernal size (same for each layer)
    ks   = data["ks"  ]
    # stride (same for each layer)
    st   = data["st"  ]
    # leaky ReLU negative slopes
    dns  = data["dns" ]
    gns  = data["gns" ]
    
    # warn about inefficient values
    warn_string = ""
    if st >= ks:
        warn_string += "No kernel overlap, image quality will suffer"+"\n"
    if not(ks%st == 0):
        warn_string += "Kernel-stride mismatch, image quality will suffer"+"\n"
    util.warn_out(warn_string)
    
    # set kernel sizes and strides for each layer
    dk = gk = [ks ]*lays
    ds = gs = [st ]*lays
    
    ### find filters
    ## construct number of i/o channels for each filter
    ##     i/o channels start at one power of 2 higher than img_size
    ##     and continue for lays-1 powers of 2
    # find nearest power of 2 for img_size (2^n = img_size ==> n)
    n_power = int(np.log(img_size)/np.log(2))
    # find lays-1 powers of 2 starting from n_power
    nc = [2**n for n in range(n_power, n_power+lays-1)]
    # discriminator channels should decrease in size,
    # generator channels should increase
    df, gf = [img_channels, *nc[::-1], 1], [z_channels, *nc, img_channels]
    
    ### find padding
    ## discriminator
    dp = util.find_padding_convolution(dk,ds,img_size,1,[0]*lays,[ks-1]*lays)
    ## generator
    # find minimum padding required by information density constraint
    gp = [np.ceil(k/s).astype(type(p)) if p<k/s else p for k,s,p in zip(gk,gs,[0]*lays)]
    # find padding required to produce a natural latent space vector
    gp = util.find_padding_deconvolution(gk,gs,img_size,gp,[ks-1]*lays)

    # Make nets
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.modules     = []
            self.kernel_size = gk
            self.stride      = gs
            self.padding     = gp
            for lay, (k,s,p) in enumerate(zip(self.kernel_size,self.stride,self.padding)):
                if n_dims == 3:
                    self.modules.append(nn.ConvTranspose3d(gf[lay], gf[lay+1], k, s, p, bias=False))
                    self.modules.append(nn.BatchNorm3d(gf[lay+1]))
                elif n_dims == 2:
                    self.modules.append(nn.ConvTranspose2d(gf[lay], gf[lay+1], k, s, p, bias=False))
                    self.modules.append(nn.BatchNorm2d(gf[lay+1]))
                # self.modules.append(nn.ReLU())
                self.modules.append(nn.LeakyReLU(gns)) ########## Testing
            #keep the last Conv, ditch the rest
            self.modules = self.modules[:-2]
            if imtype in ['grayscale', 'colour']:
                self.modules.append(nn.Tanh())
            else:
                self.modules.append(nn.Softmax())
            self.sequential = nn.Sequential(*self.modules)
        def forward(self, x):
            x = self.sequential(x)
            if imtype in ['grayscale', 'colour']:
                #normalize to [0,1]
                x = (x+1)/2
            return x
        def __str__(self):
            return(str(self.sequential))

    

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.modules     = []
            self.kernel_size = dk
            self.stride      = ds
            self.padding     = dp
            for lay, (k, s, p) in enumerate(zip(self.kernel_size,self.stride,self.padding)):
                self.modules.append(nn.Conv2d(df[lay], df[lay + 1], k, s, p, bias=False))
                # self.modules.append(nn.Conv2d   (df[lay], df[lay + 1], 1, 1, 0, bias=False))
                # self.modules.append(nn.MaxPool2d(2, 2, 0))
                # self.modules.append(nn.ReLU())
                self.modules.append(nn.LeakyReLU(dns)) ########## Testing
            #keep the last Conv, ditch the rest
            #self.modules = self.modules[:-1]
            self.sequential = nn.Sequential(*self.modules)
        def forward(self, x):
            x = self.sequential(x)
            return x
        def __str__(self):
            return(str(self.sequential))

    return Discriminator, Generator
