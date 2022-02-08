### Welcome to SliceGAN ###
####### Steve Kench #######
'''
Use this file to define your settings for a training run, or
to generate a synthetic image using a trained generator.
'''

from slicegan import preprocessing, networks, model, util
from dream3d_support import utils_json
import argparse
import json

# Run with False to show an image during or after training
parser = argparse.ArgumentParser()
parser.add_argument('training', type=int)
parser.add_argument("--inp", dest="path_input", default="inputs_3d.json", type=str, help="Specify input file location")
args       = parser.parse_args()
Training   = args.training
path_input = args.path_input

# Import user variables
with open(path_input, 'r') as f:
    data = json.load(f)["main"]
    
### SliceGAN variables

## Output structuring
# Used as folder in output directory and name for output files
Project_name       = data["Project_name"      ]
# Output directory
Project_dir        = data["Project_dir"       ]

## Data Processing
# Define image  type (colour, grayscale, three-phase or two-phase.
# n-phase materials must be segmented)
image_type         = data["image_type"        ]
# define data type (for colour/grayscale images, must be 'colour' / '
# greyscale. nphase can be, 'tif', 'png', 'jpg','array','text')
data_type          = data["data_type"         ]
# Path to your data. One string for isotrpic, 3 for anisotropic
data_path          = data["data_path"         ]
# Normalize data from color [0,255] to [0,1]
Normalize          = data["Normalize"         ]
# Scale factor vs raw data
scale_factor       = data["scale_factor"      ]

## Network Architectures
# should a 2D or 3D microstructure be created
n_dims             = data["n_dims"            ]
# Number of image channels (rbg=3, rbga=4, etc)
img_channels       = data["img_channels"      ]

### DREAM.3D support (preprocessor for ebsd scans, postprocessor for .dream3d files)
# enable DREAD.3D support? (True/False)
Dream3d_Support    = data["Dream3d_Support"   ]
# path to pipelinerunner(.exe)
dream3d_path       = data["dream3d_path"      ]
# list of names of .ctf ebsd files (or .dream3d files)
# must be in /Input directory
ebsd_paths         = data["ebsd_paths"        ]
# the suffixes of the data containers containing plane information.
# order matters! see included orientation reference.
# the order of the data containers in dream3d do not matter, only what you put here.
# suffix should not contain x,y,z except to denote plane basis.
plane_suffixes     = data["plane_suffixes"    ]
# rotation is n 90 degree counter-clockwise rotations.
# see included orientation reference to determine correct rotations
plane_rotations    = data["plane_rotations"   ]
# prefix to .tif files.
# EX: tif_names=AA7050, plane_suffixes=["YZ","XY","XY"]
# will be renamed to "AA7050_YZ.tif", "AA7050_XZ.tif", "AA7050_XY.tif"
tif_names          = data["tif_names"         ]
# valid entries are specified under dream3d_support/properties.json:
orientations_types = data["orientations_types"]
# the name of the orientation data in the .dream3d files
# only applicable if .dream3d files were specified in ebsd_paths
orientations_names = data["orientations_names"]
# plot histograms of orientations for all planes in same window
# helps determine if plane_rotations is correct.
# plane_rotations is correct if histograms overlap
plot_orientations  = data["plot_orientations" ]

## Microstructure Generation
# determines if the direction should be rendered as periodic (True/False)
# is either False to denote no periodicity
# or a list of bools with n dimensions to denote periodicity in each direction
# default is [1,1,0]
periodic           = data["periodic"          ]
# value that determines how large the output image should be
# output size is roughly: lf*32
lf                 = data["lf"                ]

# Training image size
# must be a power of 2
img_size           = data["img_size"          ]
# z vector depth
z_channels         = data["z_channels"        ]

print("Starting SliceGAN")
print(f"Using {data_type} as transfer type")
print(f"DREAM.3D support is {Dream3d_Support}")

# Create project folder
Project_path = util.mkdr(Project_name, Project_dir, Training)

if Dream3d_Support:
    import os
    from dream3d_support import preprocessor, postprocessor

    ###set up preprocessor behavior
    Normalize = True
    scale_factor = 1

    ###calculate required number of image channels from orientations_types
    properties = utils_json.pull_input(os.path.realpath("dream3d_support"+"/"+"properties.json"))
    img_channels = sum([properties[orientations_type]["n_components"] for orientations_type in orientations_types])
    
    ###set up slicegan inputs for different dream3d data types
    if img_channels > 1:
        image_type = 'colour'
        if not data_type == "text":
            data_type = 'colour'
    else:
        image_type = 'grayscale'
        if not data_type == "text":
            data_type = 'grayscale'

    ###setup paths
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # tif_path = dir_path+"/Input/"+Project_name+"/"+tif_names ######## doesn't work for optimize_parameters
    tif_path = dir_path+"/Input/"+tif_names
    
    #rewrite data_path
    data_path = []
    if data_type == "text":
        ext = ".txt"
    else:
        ext = ".tif"
    for plane_suffix in plane_suffixes:
        # data_path.append("Input/"+Project_name+"/"+tif_names+plane_suffix+".tif")   ############# doesn't work for optimize_parameters 
        data_path.append("Input/"+tif_names+plane_suffix+ext)

## Create Networks
netD, netG = networks.slicegan_nets(path_input, image_type, img_size, img_channels, z_channels, n_dims)

# Dream3d Preprocessor (converts EBSD data to SliceGAN compatible Tiff images)
if Dream3d_Support and Training:
        preprocessor.convert_ebsd_to_tiff(dir_path,Project_name,ebsd_paths,dream3d_path,tif_path,plane_suffixes,plane_rotations,data_type,orientations_types,orientations_names,n_dims,plot_orientations)

# Train
if Training:
    print('Loading Dataset...')
    datasets = preprocessing.batch(data_path, data_type, img_size, scale_factor, Normalize)
    model.train(path_input, Project_path, image_type, datasets, netD, netG, img_channels, img_size, z_channels, n_dims)

# Generate
else:
    img, raw, netG = util.test_img(Project_path, image_type, netG(), n_dims, img_channels, data_type, nz=z_channels, lf=lf, periodic=periodic)

# Dream3d Postprocessor (converts Tiff stacks into DREAM.3D files)
if Dream3d_Support and not Training:
        postprocessor.convert_slicegan_to_dream3d(dir_path, Project_dir, Project_name, plane_suffixes, dream3d_path, orientations_types, input_type=data_type)
