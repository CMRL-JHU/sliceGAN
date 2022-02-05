from slicegan import util
import torch.nn as nn
from numpy import log
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
    # padding (unique for each layer)
    dp   = data["dp"  ]
    gp   = data["gp"  ]
    # leaky ReLU negative slopes
    dns  = data["dns" ]
    gns  = data["gns" ]
    
    # warn about inefficient values
    warn_string = ""
    if st >= ks:
        warn_string += "No kernel overlap, image quality will suffer"+"\n"
    if not(ks%st == 0):
        warn_string += "Kernel-stride mismatch, image quality will suffer"+"\n"
    if gp < [ks-st]:
        warn_string += "Padding too small, image quality at edges will suffer"+"\n"
    if gp > [ks]:
        warn_string += "Padding too large, image quality at edges will suffer"+"\n"
    util.warn_out(warn_string)
    
    #set kernel sizes and strides for each layer
    dk = [ks ]*lays 
    gk = [ks ]*lays
    ds = [st ]*lays 
    gs = [st ]*lays
    
    # construct number of i/o channels for each filter
    n_power = int(log(img_size)/log(2)) # 2^n = img_size ==> n
    nc = [2**n for n in range(n_power, n_power+lays-1)]
    df, gf = [img_channels, *nc, 1], [z_channels, *nc[::-1], img_channels]
    
    #find discriminator padding
    dp = util.find_padding(ks, st, df)
    
    #find minimum image size
    img_size_min = util.find_min_input_size_convolution(dk,ds,dp,"image_size",1)
    if img_size < img_size_min:
        raise ValueError("Image size (%(img_size)s) is less than minimum image size (%(img_size_min)s). Cannot convolve." %{"img_size":img_size, "img_size_min":img_size_min})
    
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
                # Convolve and rectify the input image
                self.modules.append(nn.Conv2d(df[lay], df[lay + 1], k, s, p, bias=False))
                self.modules.append(nn.LeakyReLU(dns))
            
            # Previous will output [nc x 1] tensor.
            # Flatten to ready for Linear
            # Linear to learn from each of the (nc) outputs
            # self.modules.append(nn.Flatten(start_dim=1, end_dim=-1))
            # self.modules.append(nn.Linear(img_channels, 1))
            
            # Combine the modules into a sequential network
            self.sequential = nn.Sequential(*self.modules)
        def forward(self, x):
            x = self.sequential(x)
            return x
        def __str__(self):
            return(str(self.sequential))

    return Discriminator, Generator
