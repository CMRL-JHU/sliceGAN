import os
from torch import nn
import torch
from torch import autograd
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import sys
import itertools

from warnings import warn
import time
import sys

force_cpu = True

## Training Utils

def mkdr(proj,proj_dir,Training):
    """
    When training, creates a new project directory or overwrites an existing directory according to user input. When testing, returns the full project path
    :param proj: project name
    :param proj_dir: project directory
    :param Training: whether new training run or testing image
    :return: full project path
    """
    pth = proj_dir + '/' + proj
    if Training:
        try:
            # os.mkdir(pth)
            os.makedirs(pth)
            return pth + '/' + proj
        except FileExistsError:
            print('Directory', pth, 'already exists. Enter new project name or hit enter to overwrite')
            new = input()
            if new == '':
                return pth + '/' + proj
            else:
                pth = mkdr(new, proj_dir, Training)
                return pth
        # except FileNotFoundError:
        #     print('The specifified project directory ' + proj_dir + ' does not exist. Please change to a directory that does exist and again')
        #     sys.exit()
    else:
        return pth + '/' + proj

def weights_init(m):
    """
    Initialises training weights
    :param m: Convolution to be intialised
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, gp_lambda, nc):
    """
    calculate gradient penalty for a batch of real and fake data
    :param netD: Discriminator network
    :param real_data:
    :param fake_data:
    :param batch_size:
    :param l: image size
    :param device:
    :param gp_lambda: learning parameter for GP
    :param nc: channels
    :return: gradient penalty
    """
    
    #sample and reshape random numbers
    alpha = torch.rand(batch_size, 1, device = device)
    alpha = alpha.expand(-1, real_data.nelement()//batch_size).contiguous()
    alpha = alpha.view(-1, nc, l, l)
    
    #create interpolate dataset
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates.requires_grad_(True)

    #pass interpolates through netD
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device = device),
                              create_graph=True, only_inputs=True)[0]
    # extract the grads and calculate gp
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


def calc_eta(steps, time, start, i, epoch, num_epochs):
    """
    Estimates the time remaining based on the elapsed time and epochs
    :param steps:
    :param time: current time
    :param start: start time
    :param i: iteration through this epoch
    :param epoch:
    :param num_epochs: totale no. of epochs
    """
    elap = time - start
    progress = epoch * steps + i + 1
    rem = num_epochs * steps - progress
    ETA = rem / progress * elap
    hrs = int(ETA / 3600)
    mins = int((ETA / 3600 % 1) * 60)
    print('[%d/%d][%d/%d]\tETA: %d hrs %d mins'
          % (epoch, num_epochs, i, steps,
             hrs, mins))

## Plotting Utils
def post_proc(img,imtype):
    """
    turns one hot image back into grayscale
    :param img: input image
    :param imtype: image type
    :return: plottable image in the same form as the training data
    """
    try:
        #make sure it's one the cpu and detached from grads for plotting purposes
        img = img.detach().cpu()
    except:
        pass
    # for n phase materials, seperate out the channels and take the max
    if imtype == 'twophase':
        img_pp = np.zeros(img.shape[2:])
        p1 = np.array(img[0][0])
        p2 = np.array(img[0][1])
        img_pp[(p1 < p2)] = 1  # background, yellow
        return img
    if imtype == 'threephase':
        img_pp = np.zeros(img.shape[2:])
        p1 = np.array(img[0][0])
        p2 = np.array(img[0][1])
        p3 = np.array(img[0][2])
        img_pp[(p1 > p2) & (p1 > p3)] = 0  # background, yellow
        img_pp[(p2 > p1) & (p2 > p3)] = 1  # spheres, green
        img_pp[(p3 > p2) & (p3 > p1)] = 2  # binder, purple
        return img_pp
    # colour and grayscale don't require post proc, just a shift
    if imtype == 'colour':
        return np.int_(255 * (np.swapaxes(img[0], 0, -1)))
    if imtype == 'grayscale':
        return 255*img[0][0]

def test_plotter(img,imtype,pth,n_dims=3,slcs=5):
    """
    creates a fig with 3*slc subplots showing example slices along the three axes
    :param img: raw input image
    :param slcs: number of slices to take in each dir
    :param imtype: image type
    :param pth: where to save plot
    """

    img = post_proc(img,imtype)
    
    if n_dims == 2:
        if imtype == 'colour':
            plt.imshow(img, vmin = 0, vmax = 255)
        elif imtype == 'grayscale':
            plt.imshow(img, cmap = 'gray')
        else:
            plt.imshow(img)
    
    if n_dims == 3:
        fig, axs = plt.subplots(slcs, 3)
        if imtype == 'colour':
            for j in range(slcs):
                axs[j, 0].imshow(img[j, :, :, :], vmin = 0, vmax = 255)
                axs[j, 1].imshow(img[:, j, :, :], vmin = 0, vmax = 255)
                axs[j, 2].imshow(img[:, :, j, :], vmin = 0, vmax = 255)
        elif imtype == 'grayscale':
            for j in range(slcs):
                axs[j, 0].imshow(img[j, :, :], cmap = 'gray')
                axs[j, 1].imshow(img[:, j, :], cmap = 'gray')
                axs[j, 2].imshow(img[:, :, j], cmap = 'gray')
        else:
            for j in range(slcs):
                axs[j, 0].imshow(img[j, :, :])
                axs[j, 1].imshow(img[:, j, :])
                axs[j, 2].imshow(img[:, :, j])
    
    plt.savefig(pth + '_slices.png')
    plt.close()

def permute(*strings): #used in plotting loss graphs if not splitting graphs
    strings = list(itertools.product(*strings))
    strings = [' '.join(string) for string in strings]
    return strings

def graph_plot(data,labels,pth,name):
    """
    simple plotter for all the different graphs
    :param data: a list of data arrays
    :param labels: a list of plot labels
    :param pth: where to save plots
    :param name: name of the plot figure
    :return:
    """

    for datum,lbl in zip(data,labels):
        plt.plot(datum, label = lbl)
    plt.legend()
    plt.savefig(pth + '_' + name + ".png")
    plt.close()


def test_img(pth, imtype, netG, n_dims, nc, output_type, nz = 64, lf = 4, periodic=False):
    """
    saves a test volume for a trained or in progress of training generator
    :param pth: where to save image and also where to find the generator
    :param imtype: image type
    :param netG: Loaded generator class
    :param nz: latent z dimension
    :param lf: length factor
    :param show:
    :param periodic: list of periodicity in axis 1 through n
    :return:
    """

    # force the generator to use the CPU rather than whatever was used to generate the data
    if force_cpu:
        netG.load_state_dict(torch.load(pth + '_Gen.pt', map_location=torch.device('cpu')))
    else:
        netG.load_state_dict(torch.load(pth + '_Gen.pt'))
        
    netG.eval()
    noise = torch.randn(1, nz, *[lf]*n_dims)
    if periodic:
        if periodic[0]:
            noise[:, :, :2] = noise[:, :, -2:]
        if periodic[1]:
            noise[:, :, :, :2] = noise[:, :, :, -2:]
        if periodic[2]:
            noise[:, :, :, :, :2] = noise[:, :, :, :, -2:]
    raw = netG(noise)
    print('Postprocessing')
    gb = post_proc(raw,imtype)
    if periodic:
        if periodic[0]:
            gb = gb[:-1]
        if periodic[1]:
            gb = gb[:,:-1]
        if periodic[2]:
            gb = gb[:,:,:-1]
    tif = np.int_(gb)
    
    # find shape data
    shape = list(tif.shape)
    if n_dims == 2:
        shape = [1]+shape
    if nc == 1:
        shape = shape+[1]
    tif = tif.reshape(tuple(shape))

    # write out to txt file
    if output_type == "text":
        array_to_text(tif, pth + '.txt')
    
    # write out to tif file
    else:
        if shape[-1] <= 4:
            tifffile.imwrite(pth + '.tif', tif)
        else:
            warn_out("Too many components for tiff image. No image will be created. Use text output instead.")
        
    return tif, raw, netG
    
def array_to_text(data, path_output, number_format='%.0f', delimiter="\t"):
    dims         = list(data.shape[:-1][::-1])
    n_components = data.shape[-1 ]
        
    # break apart the list into components
    shape = (np.prod(dims), n_components)
    data = data.reshape(shape)
    
    with open(path_output, "w") as file_output:
        file_output.write(delimiter.join(map(str, dims[::-1]+[n_components]))+"\n")
    with open(path_output, "ab") as file_output:
        np.savetxt(file_output, data, fmt=number_format, delimiter=delimiter)

def find_min_input_size_convolution(k,s,p,name,size_current,silent=False):
    # for i in range(len(k)-1,0-1,-1):
    for k,s,p in zip(k[::-1],s[::-1],p[::-1]):
        # size_next = (size_current-1)*s[i]+k[i]-2*p[i]
        size_next = (size_current-1)*s+k-2*p
        size_current=size_next
    if not(size_next%1 == 0):
        if silent:
            return False
        else:
            raise ValueError("Malformed kernel size or stride caused non-integer value for %(name)s. Cannot resolve." %{"name":name})
    return int(size_next)

def find_min_input_size_deconvolution(k,s,p,name,size_current,silent=False):
    # for i in range(len(k)-1,0-1,-1):
    for k,s,p in zip(k[::-1],s[::-1],p[::-1]):
        # size_next = (size_current-k[i]+2*p[i])/s[i]+1
        size_next = (size_current-k+2*p)/s+1
        size_current=size_next
    if not(size_next%1 == 0):
        if silent:
            return False
        else:
            raise ValueError("Malformed kernel size or stride caused non-integer value for %(name)s. Cannot resolve." %{"name":name})
    return int(size_next)
    
def find_padding_deconvolution(k,s,output_size,bounds_min,bounds_max):
    # raise the padding values by:
    # counting from right to left,
    # treating each convolution as a digit,
    # bounding each digit between [p[i],bounds[i]]
    for p in find_permutations_between_bounds(bounds_min,bounds_max):
        if find_min_input_size_deconvolution(k,s,p,"lz",output_size,silent=True):
            return p
    raise ValueError("Could not find value for padding that would result in natural input size")
    
def find_padding_convolution(k,s,input_size,output_size,bounds_min,bounds_max):
    # raise the padding values by:
    # counting from right to left,
    # treating each convolution as a digit,
    # bounding each digit between [p[i],bounds[i]]
    for p in find_permutations_between_bounds(bounds_min,bounds_max):
        if find_min_input_size_convolution(k,s,p,"lz",output_size,silent=True) == input_size:
            return p
    raise ValueError(f"Could not find value for padding that would result in output size of {input_size}")
    
def find_permutations_between_bounds(bounds_min,bounds_max):
    ps = []
    for i in range(bounds_min[0], max(bounds_min[0],bounds_max[0]+1)):
        if len(bounds_min) > 1:
            for j in find_permutations_between_bounds(bounds_min[1:],bounds_max[1:]):
                ps += [[i]+j]
        else:
            ps += [[i]]
    return ps

def warn_out(warning_string):
    if warning_string:
        print(warning_string)
        pause_for(3)

def pause_for(s):
    for i in range(1,3+1)[::-1]:
        sys.stdout.write("\rContinuing in... %i" % i)
        sys.stdout.flush()
        time.sleep(1)
    print("")
    
def import_text_file(path_text, delineator="\t"):
    with open(path_text) as file_input:
        shape, string_input = file_input.read().split("\n", 1)
    shape         = extract_numbers(shape       , delineator=delineator, value_type=int).tolist()[0]
    numbers_input = extract_numbers(string_input, delineator=delineator)
    return shape, numbers_input
    
def extract_numbers(string, delineator="\t", value_type=float):
    numbers_matrix = []
    for i, sentence in enumerate(string.splitlines()):
        numbers_line = []
        for j, word in enumerate(sentence.split(delineator)):
            if is_number(word):
                numbers_line.append(value_type(word))
        if len(numbers_line) > 0:
            numbers_matrix.append(numbers_line)
    return(np.asarray(numbers_matrix))

def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
