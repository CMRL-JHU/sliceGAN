import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

file_input_path_relative = "/pipeline_output/5-comparestats.dream3d"
view_mode = "separate"#"same"
domains = ["Synthetic","EBSD"]
data_paths = ["CellFeatureData/MisorientationList","CellFeatureData/EquivalentDiameters","CellFeatureData/AspectRatios"]
headers = ["Feature Neighbor Misorientations (Degrees)","Equivalent Diameters (micrometers)","AspectRatios"]
xlabels = ["Neighbor Misorientations (Degrees)","Equivalent Diameters (micrometers)","AspectRatios"]
titles  = [["misorientation"],["Equivalent Diameters"],["AspectRatios b/a","AspectRatios c/a"]]
bins = 50
alpha=.5

dir_path = os.path.dirname(os.path.realpath(__file__))
file_input_path_absolute = dir_path+file_input_path_relative

def set_headers(axes,rows=None,cols=None,pad=5):
    if not rows is None:
        for ax, row in zip(axes[:,0], rows):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center', rotation=90)
    if not cols is None:
        for ax, col in zip(axes[0], cols):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline', rotation=90)
    #subplots are pointers, no need for a return
def plot_histogram(axes,data,titles=None,label=None,xlabel=None,bins=40,alpha=.5):
    shape = list(data.shape)
    if len(shape) > 2:
        data = data.reshape(np.prod(shape[:-1]),shape[-1])
    elif len(shape) < 2:
        data = data.reshape(shape[0],1)
    shape = list(data.shape)
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
        # axes[i].set_ylabel("Frequency (Norm)")
    #subplots are pointers, no need for a return
def find_data_path(DataContainer,data_path):
    path, _, name = data_path.rpartition("/")
    path = list(filter(lambda x: x.startswith(path), DataContainer.keys()))[0]
    return path+"/"+name
    
def plot_same(domains,data_paths,headers,xlabels,titles,bins,alpha):
    dims    = [len(titles)*len(domains),max(map(len, titles))]                    
    DataContainers = []
    for DataContainer in file_input["DataContainers"]:
        DataContainers.append(file_input["DataContainers"+"/"+DataContainer])
    fig, axes = plt.subplots(nrows=dims[0],ncols=dims[1],sharex="row")
    for DataContainer in DataContainers:
        name_plane = DataContainer.name.replace("/DataContainers/ImageDataContainer_","")
        for i in range(len(data_paths)):
            data = DataContainer[find_data_path(DataContainer,data_paths[i])][...]
            domain_indecies = [index for index, domain in enumerate(domains) if domain in DataContainer.name]
            for domain_index in domain_indecies:
                plot_histogram(axes[i*len(domains)+domain_index],data,titles[i],name_plane,xlabels[i],bins=bins,alpha=alpha)
    set_headers(axes,rows=headers)
    # fig.tight_layout()
    plt.show()

def plot_separate(domains,data_paths,headers,xlabels,titles,bins,alpha):
    DataContainers = []
    for DataContainer in file_input["DataContainers"]:
        DataContainers.append(file_input["DataContainers"+"/"+DataContainer])
    for i in range(len(data_paths)):
        dims    = [len(domains),len(titles[i])]
        fig, axes = plt.subplots(nrows=dims[0],ncols=dims[1],sharex="row")
        axes = np.array(axes).reshape(tuple(dims))
        for DataContainer in DataContainers:
            name_plane = DataContainer.name.replace("/DataContainers/ImageDataContainer_","")
            data = DataContainer[find_data_path(DataContainer,data_paths[i])][...]
            ########################################################### Temporary
            # if "AspectRatios" in headers[i]:
                # data = data[:,0]
            #####################################################################
            domain_indecies = [index for index, domain in enumerate(domains) if domain in DataContainer.name]
            for domain_index in domain_indecies:
                print(name_plane)
                plot_histogram(axes[domain_index],data,titles[i],name_plane,None,bins=bins,alpha=alpha)
                ####################################################### Temporary
                # if "Equivalent Diameters" in headers[i]:
                    # axes[domain_index][0].set_xlim([0,3])
                # if "AspectRatios" in headers[i]:
                    # axes[domain_index][0].set_xlim([0,1])
                    # axes[domain_index][0].set_ylim([0,.2])
                #################################################################
        headers_current = [domains[j]+" Domain" for j in range(len(domains))]
        set_headers(axes,rows=headers_current)
        fig.suptitle(headers[i]+" vs Frequency (Normalized)")
        # fig.tight_layout()
        plt.show()
        
with h5py.File(file_input_path_absolute, 'r') as file_input:
    if view_mode == "same":
        plot_same(domains,data_paths,headers,xlabels,titles,bins,alpha)
    elif view_mode == "separate":
        plot_separate(domains,data_paths,headers,xlabels,titles,bins,alpha)