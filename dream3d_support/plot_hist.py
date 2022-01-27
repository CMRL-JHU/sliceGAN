import h5py, tifffile, os, sys
import numpy as np
import matplotlib.pyplot as plt
new_path = os.path.dirname(__file__)
if new_path not in sys.path:
    sys.path.append(new_path)
import utils_dream3d

#create histograms for the orientations to ensure they're plotting in an expected manner
def plot_orientations_hist(orientation_type,DataContainers,name_planes,name_images):
    if orientation_type == "Quats":
        data = []
        for i in range(len(name_images)):
            data.append([
                utils_dream3d.remove_empty_z_container(DataContainers[i]["CellData/Quats"][:]),
                (utils_dream3d.remove_empty_z_container(DataContainers[i]["CellData/Quats"][:])+1)*127.5,
                tifffile.imread(name_images[i]),
                tifffile.imread(name_images[i])/127.5-1
                ])
        plot_data_hist(
            data,
            ["Quats","Quats Scaled","Quats From File","Quats From File Unscaled"],
            ["Orientations [-1,1]","Orientations [0,255]","Orientations [0,255]","Orientations [-1,1]"],
            [["a","b","c","d"],["a","b","c","d"],["a","b","c","d"],["a","b","c","d"]],
            name_planes)
    elif orientation_type == "EulerAngles":
        data = []
        for i in range(len(name_images)):
            data.append([
                np.append(
                    utils_dream3d.remove_empty_z_container(DataContainers[i]["CellData/EulerAngles"][:]),
                    utils_dream3d.remove_empty_z_container(DataContainers[i]["CellData/Phases"][:]),
                    2),
                np.append(
                    utils_dream3d.remove_empty_z_container(DataContainers[i]["CellData/EulerAngles"][:])/(2*np.pi)*255,
                    utils_dream3d.remove_empty_z_container(DataContainers[i]["CellData/Phases"][:])/DataContainers[i]["CellData/Phases"][:].max()*255,
                    2),
                tifffile.imread(name_images[i]),
                np.append(
                    tifffile.imread(name_images[i])[:,:,:-1]/255*2*np.pi,
                    tifffile.imread(name_images[i])[:,:,-1].reshape(tuple(np.append(tifffile.imread(name_images[i])[:,:,-1].shape,1)))/255*DataContainers[i]["CellData/Phases"][:].max(),
                    2)
                ])
        plot_data_hist(
            data,
            ["Euler Angles","Euler Angles Scaled","Euler Angles From File","Euler Angles From File Unscaled"],
            ["Orientations (rads)","Orientations [0,255]","Orientations [0,255]","Orientations (rads)"],
            [["α","β","γ","Phase ID"],["α","β","γ","Phase ID"],["α","β","γ","Phase ID"],["α","β","γ","Phase ID"]],
            name_planes)
def plot_data_hist(data,headers,xlabels,titles,name_planes,bins=40,alpha=.5):
    import matplotlib.pyplot as plt
    def set_headers(axes,rows=None,cols=None,pad=5):
        if not rows is None:
            for ax, row in zip(axes[:,0], rows):
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points',
                            size='large', ha='right', va='center')
        if not cols is None:
            for ax, col in zip(axes[0], cols):
                ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                            xycoords='axes fraction', textcoords='offset points',
                            size='large', ha='center', va='baseline')
        #subplots are pointers, no need for a return
    def plot_histogram(axes,data,titles=None,label=None,xlabel=None,bins=40,alpha=.5):
        shape = list(data.shape)
        data = data.reshape(np.prod(shape[:-1]),shape[-1])
        weights = np.ones_like(data)/len(data)
        for i in range(shape[-1]):
            if not label is None:
                axes[i].hist(x=data[:,i],bins=bins,weights=weights[:,i],label=label,alpha=alpha)
                axes[i].legend()
            else:
                axes[i].hist(x=data[:,i],bins=bins,weights=weights[:,i],alpha=alpha)
            if not titles is None:
                axes[i].title.set_text(titles[i])
            if not xlabel is None:
                axes[i].set_xlabel(xlabel)
            axes[i].set_ylabel("Frequency (Norm)")
        #subplots are pointers, no need for a return                            
    dims    = [len(titles),max(map(len, titles))]
    fig, axes = plt.subplots(nrows=dims[0],ncols=dims[1],sharex="row")
    for i in range(len(data)):
        for j in range(len(data[i])):
            plot_histogram(axes[j,:],data[i][j],titles[j],name_planes[i],xlabels[j],bins=bins,alpha=alpha)
    set_headers(axes,rows=headers)
    # fig.tight_layout()
    plt.show()