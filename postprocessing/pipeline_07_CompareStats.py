import h5py
import numpy as np
import matplotlib.pyplot as plt

output_path = "pipeline_output/07-compare_statistics"

data = {
    "files":{
        "Prior Particles":{
            "EBSD YZ Small Plane":"pipeline_input/6-yz_small_cleaned_grains_feature_attributes.dream3d",
            "Synthetic":"pipeline_output/6-feature_attributes.dream3d"
        }
    },
    "statistics":{
        "MisorientationList":{
            "path"   :"DataContainers/ImageDataContainer/CellFeatureData/MisorientationList",
            "header" :"Feature Neighbor Misorientations (Degrees)",
            "xlabel" :"Feature Neighbor Misorientations (Degrees)",
            "titles" :["Misorientation"],
            "figsize":(5,5),
            "xlim"   :[0, 65],
            "ylim"   :None,
            "bins"   :50
        },
        "EquivalentDiameters":{
            "path"   :"DataContainers/ImageDataContainer/CellFeatureData/EquivalentDiameters",
            "header" :"Equivalent Diameters (micrometers)",
            "xlabel" :"Equivalent Diameters (micrometers)",
            "titles" :["Equivalent Diameters"],
            "figsize":(5,5),
            "xlim"   :[0, 10],
            "ylim"   :None,
            "bins"   :50
        },
        "Volumes":{
            "path"   :"DataContainers/ImageDataContainer/CellFeatureData/Volumes",
            "header" :"Volumes (micrometers)",
            "xlabel" :"Volumes (micrometers)",
            "titles" :["Volumes"],
            "figsize":(5,5),
            "xlim"   :[0, 10],
            "ylim"   :None,
            "bins"   :50
        },
        "AspectRatios":{
            "path"   :"DataContainers/ImageDataContainer/CellFeatureData/AspectRatios",
            "header" :"AspectRatios",
            "xlabel" :"AspectRatios",
            "titles" :["AspectRatios b/a", "AspectRatios c/a"],
            "figsize":(10,5),
            "xlim"   :[0.2, 1],
            "ylim"   :None,
            "bins"   :20
        }
    }
}

alpha = 0.5
save  = True

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
    
def plot_histogram(axes,data,titles=None,label=None,xlabel=None,xlim=None,ylim=None,bins=40,alpha=.5):
    shape = list(data.shape)

    # flatten dimensions, leave component dimension
    if len(shape) > 2:
        data = data.reshape(np.prod(shape[:-1]),shape[-1])
    # add component dimension
    elif len(shape) < 2:
        data = data.reshape(shape[0],1)
        
    # weights are used to normalize histogram
    weights = np.ones_like(data)/len(data)
    
    for axis, title, weights_axis, data_axis in zip(axes, titles, weights.T, data.T):
        HIST_BINS = np.linspace(xlim[0], xlim[1], bins)
        if not label is None:
#            axis.hist(x=data_axis,bins=bins,weights=weights_axis,label=label,alpha=alpha)
            axis.hist(
                x=data_axis,
                bins=HIST_BINS,
                weights=weights_axis,
                label=label,
                alpha=alpha,
                edgecolor='black',
                linewidth=1.2
                )
            axis.legend()
        else:
#            axis.hist(x=data_axis,bins=bins,weights=weights_axis,alpha=alpha)
            axis.hist(x=data_axis,bins=HIST_BINS,weights=weights_axis,alpha=alpha)
        if not titles is None:
            axis.title.set_text(title)
        if not xlabel is None:
            axis.set_xlabel(xlabel)
        if not xlim is None:
            axis.set_xlim(xlim)
        if not ylim is None:
            axis.set_ylim(ylim)
        # axes[i].set_ylabel("Frequency (Norm)")
    #subplots are pointers, no need for a return

def compare_statistics(inputs,alpha,save):

    domains = [key for key in inputs["files"].keys()]
    for domain in domains:
    
        # open all *.dream3d files in the current domain
        input_files = [h5py.File(path,'r') for path in [path for path in inputs["files"][domain].values()]]
        input_names = [name for name in inputs["files"][domain].keys()]

        statistics = [statistic for statistic in inputs["statistics"].keys()]
        for statistic in statistics:
        
            path    = inputs["statistics"][statistic]["path"   ]
            header  = inputs["statistics"][statistic]["header" ]
            xlabel  = inputs["statistics"][statistic]["xlabel" ]
            titles  = inputs["statistics"][statistic]["titles" ]
            figsize = inputs["statistics"][statistic]["figsize"]
            xlim    = inputs["statistics"][statistic]["xlim"   ]
            ylim    = inputs["statistics"][statistic]["ylim"   ]
            bins    = inputs["statistics"][statistic]["bins"   ]
        
            # set up subplots (n_domains x n_titles)
            dims    = (len(domains),len(titles))
            fig, axes = plt.subplots(nrows=dims[0],ncols=dims[1],sharex="row",figsize=figsize)
            axes = np.array(axes).reshape(tuple(dims))
            
            for input_file, input_name in zip(input_files, input_names):

                data = input_file[path][...]
                axis = axes[domains.index(domain)]
                
                plot_histogram(axis,data,titles,input_name,xlabel,xlim=xlim,ylim=ylim,bins=bins,alpha=alpha)
            
            # display the domain headers (left most label)
            set_headers(axes,rows=[domain+" Domain" for domain in domains])
            
            # rotate ticks
            for axis in axes.flatten():
                for label in axis.get_yticklabels():
                    label.set_rotation(45)
                for label in axis.get_xticklabels():
                    label.set_rotation(45)
            
            # display the statistic header (top most label)
            fig.suptitle(header+"\n"+"vs"+"\n"+"Frequency (Normalized)")
            
            # remove most whitespace
            fig.tight_layout()
            
            if save:
                plt.savefig(output_path+"_"+statistic+".png", dpi=1200)
            else:
                plt.show()
            
        [input_file.close() for input_file in input_files]

compare_statistics(data,alpha,save)
