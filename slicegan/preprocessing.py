import numpy as np
import torch
import matplotlib.pyplot as plt
import tifffile
import os, sys
new_path = os.path.dirname(__file__)
if new_path not in sys.path:
    sys.path.append(new_path)
import util
def batch(data, data_type, l, sf, Normalize=True, Testing=False):
    #l is sample image size
    #data is the path to the image file

    """
    Generate a batch of images randomly sampled from a training microstructure
    :param data: data path
    :param data_type: data type
    :param l: image size
    :param sf: scale factor
    :return:
    """
    
    ##################### need to apply the loop over channels in color to everything else ################################
    
    padding = "   "
    
    if data_type == 'png' or data_type == 'jpg':
        datasetxyz = []
        for img in data:
            print(padding+img)
            img = tifffile.imread(img)
            # Normalize values between 0 and 1
            if Normalize:
                img = img[:]/255
            if len(img.shape)>2:
                img = img[:,:,0]
            img = img[::sf,::sf]
            x_max, y_max= img.shape[:]
            phases = np.unique(img)
            data = np.empty([32 * 900, len(phases), l, l])
            for i in range(32 * 900):
                x = np.random.randint(1, x_max - l-1)
                y = np.random.randint(1, y_max - l-1)
                # create one channel per phase for one hot encoding
                for cnt, phs in enumerate(phases):
                    img1 = np.zeros([l, l])
                    img1[img[x:x + l, y:y + l] == phs] = 1
                    data[i, cnt, :, :] = img1

            if Testing:
                for j in range(7):
                    plt.imshow(data[j, 0, :, :]+2*data[j, 1, :, :])
                    plt.pause(0.3)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)

    elif data_type=='tif':
        datasetxyz=[]
        print(padding+img)
        img = np.array(tifffile.imread(data[0]))
        # Normalize values between 0 and 1
        if Normalize:
            img = img[:]/255
        img = img[::sf,::sf,::sf]
        ## Create a data store and add random samples from the full image
        x_max, y_max, z_max = img.shape[:]
        print('training image shape: ', img.shape)
        vals = np.unique(img)
        for dim in range(3):
            data = np.empty([32 * 900, len(vals), l, l])
            print('dataset ', dim)
            for i in range(32*900):
                # string = (
                # "z_max = "+str(z_max)+"\n"+
                # "l = "+str(l)+"\n"
                # )
                # print(string)###########################################################
                x = np.random.randint(0, x_max - l)
                y = np.random.randint(0, y_max - l)
                z = np.random.randint(0, z_max - l)
                # create one channel per phase for one hot encoding
                lay = np.random.randint(img.shape[dim]-1)
                for cnt,phs in enumerate(list(vals)):
                    img1 = np.zeros([l,l])
                    if dim==0:
                        img1[img[lay, y:y + l, z:z + l] == phs] = 1
                    elif dim==1:
                        img1[img[x:x + l,lay, z:z + l] == phs] = 1
                    else:
                        img1[img[x:x + l, y:y + l,lay] == phs] = 1
                    data[i, cnt, :, :] = img1[:,:]
                    # data[i, (cnt+1)%3, :, :] = img1[:,:]

            if Testing:
                for j in range(2):
                    plt.imshow(data[j, 0, :, :] + 2 * data[j, 1, :, :])
                    plt.pause(1)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)

    elif data_type=='colour':
        ## Create a data store and add random samples from the full image
        datasetxyz = []
        for img in data:
            print(padding+img)
            img = tifffile.imread(img)
            #temp
            #plt.imshow(img.astype(np.double)/img.max().astype(np.double)*255, cmap='gray', vmin=0, vmax=255)
            #plt.savefig("meh")
            # Normalize values between 0 and 1
            if Normalize:
                img = img[:]/255
            img = img[::sf,::sf,:] #determines the order (negatives reverse order) and stride (1=skip none, 2=skip one, etc...)
            ep_sz = 32 * 900 #batch size ??? 32 bit color depth maybe, but 900 ???
            #image data comes in [x,y,n_channels] where channels can be [r,b,g]=3, [r,b,g,alpha]=4
            #so this list is size: [???, n_channels, x_max, y_max]
            data = np.empty([ep_sz, list(img.shape)[2], l, l])
            x_max, y_max = img.shape[:2]
            for i in range(ep_sz):
                x = np.random.randint(0, x_max - l)
                y = np.random.randint(0, y_max - l)
                # create one channel per phase for one hot encoding
                for j in range(list(data.shape)[1]):
                    data[i, j, :, :] = img[x:x + l, y:y + l,j]
            print('converting')
            if Testing:
                datatest = np.swapaxes(data,1,3)
                datatest = np.swapaxes(datatest,1,2)
                for j in range(5):
                    plt.imshow(datatest[j, :, :, :])
                    plt.pause(0.5)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)

    elif data_type=='grayscale':
        datasetxyz = []
        for img in data:
            print(padding+img)
            img = tifffile.imread(img)
            # Normalize values between 0 and 1
            if Normalize:
                img = img[:]/255
            if len(img.shape) > 2:
                img = img[:, :, 0]
            img = img/img.max()
            img = img[::sf, ::sf]
            x_max, y_max = img.shape[:]
            data = np.empty([32 * 900, 1, l, l])
            for i in range(32 * 900):
                x = np.random.randint(1, x_max - l - 1)
                y = np.random.randint(1, y_max - l - 1)
                subim = img[x:x + l, y:y + l]
                data[i, 0, :, :] = subim
            if Testing:
                for j in range(7):
                    plt.imshow(data[j, 0, :, :])
                    plt.pause(0.3)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)
    
    elif data_type=='text':
        ## Create a data store and add random samples from the full image
        datasetxyz = []
        for img in data:
            print(padding+img)
            shape, img = util.import_text_file(img)
            img = img.reshape(shape)
            #temp
            #plt.imshow(img.astype(np.double)/img.max().astype(np.double)*255, cmap='gray', vmin=0, vmax=255)
            #plt.savefig("meh")
            # Normalize values between 0 and 1
            if Normalize:
                img = img[:]/255
            img = img[::sf,::sf,:] #determines the order (negatives reverse order) and stride (1=skip none, 2=skip one, etc...)
            ep_sz = 32 * 900 #batch size ??? 32 bit color depth maybe, but 900 ???
            #image data comes in [x,y,n_channels] where channels can be [r,b,g]=3, [r,b,g,alpha]=4
            #so this list is size: [???, n_channels, x_max, y_max]
            data = np.empty([ep_sz, list(img.shape)[2], l, l])
            x_max, y_max = img.shape[:2]
            for i in range(ep_sz):
                x = np.random.randint(0, x_max - l)
                y = np.random.randint(0, y_max - l)
                # create one channel per phase for one hot encoding
                for j in range(list(data.shape)[1]):
                    data[i, j, :, :] = img[x:x + l, y:y + l,j]
            print('converting')
            if Testing:
                datatest = np.swapaxes(data,1,3)
                datatest = np.swapaxes(datatest,1,2)
                for j in range(5):
                    plt.imshow(datatest[j, :, :, :])
                    plt.pause(0.5)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)
    
    return datasetxyz


