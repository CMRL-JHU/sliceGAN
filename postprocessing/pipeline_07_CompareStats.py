import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

output_path = "pipeline_output/07-compare_statistics.png"

data = {
    "statistics":{
        "MisorientationList":{
            "paths":{
                "EBSD":{
                    "file": "pipeline_input/6-yz_small_cleaned_grains_feature_attributes.dream3d",
                    "hdf5": "DataContainers/ImageDataContainer/CellFeatureData/MisorientationList"
                },
                "Synthetic": {
                    "file": "pipeline_output/6-feature_attributes.dream3d",
                    "hdf5": "DataContainers/ImageDataContainer/CellFeatureData/MisorientationList"
                }
            },
            "xlabels":[r"Misorientation [$^{\circ}$]"],
            "xlim"   :[0, 65],
            "ylim"   :None,
            "n_bins" :25
        },
        "EquivalentDiameters":{
            "paths":{
                "EBSD":{
                    "file": "pipeline_input/6-yz_small_cleaned_grains_feature_attributes.dream3d",
                    "hdf5": "DataContainers/ImageDataContainer/CellFeatureData/EquivalentDiameters"
                },
                "Synthetic": {
                    "file": "pipeline_output/6-feature_attributes.dream3d",
                    "hdf5": "DataContainers/ImageDataContainer/CellFeatureData/EquivalentDiameters"
                }
            },
            "xlabels":[r"Equivalent Diameters [$\mu m$]"],
            "xlim"   :[0, None],
            "ylim"   :None,
            "n_bins" :25
        },
        "Volumes":{
            "paths":{
                "EBSD":{
                    "file": "pipeline_input/6-yz_small_cleaned_grains_feature_attributes.dream3d",
                    "hdf5": "DataContainers/ImageDataContainer/CellFeatureData/Volumes"
                },
                "Synthetic": {
                    "file": "pipeline_output/6-feature_attributes.dream3d",
                    "hdf5": "DataContainers/ImageDataContainer/CellFeatureData/Volumes"
                }
            },
            "xlabels" :[r"Volumes [${\mu m}^3$]"],
            "xlim"   :[0, None],
            "ylim"   :None,
            "n_bins" :25
        },
        # "AspectRatios":{
        #     "paths":{
        #         "EBSD":{
        #             "file": "pipeline_input/6-yz_small_cleaned_grains_feature_attributes.dream3d",
        #             "hdf5": "DataContainers/ImageDataContainer/CellFeatureData/AspectRatios"
        #         },
        #         "Synthetic": {
        #             "file": "pipeline_output/6-feature_attributes.dream3d",
        #             "hdf5": "DataContainers/ImageDataContainer/CellFeatureData/AspectRatios"
        #         }
        #     },
        #     "xlabels":["Aspect Ratios b/a", "Aspect Ratios c/a"],
        #     "xlim"   :[0, 1],
        #     "ylim"   :None,
        #     "n_bins" :25
        # }
        "AspectRatios":{
            "paths":{
                "EBSD":{
                    "file": "pipeline_input/6-yz_small_cleaned_grains_feature_attributes.dream3d",
                    "hdf5": "DataContainers/ImageDataContainer/CellFeatureData/AspectRatios"
                },
                "Synthetic": {
                    "file": "pipeline_output/6-feature_attributes.dream3d",
                    "hdf5": "DataContainers/ImageDataContainer/CellFeatureData/AspectRatios"
                }
            },
            "xlabels":["Aspect Ratios"],
            "xlim"   :[0.0, 1],
            "ylim"   :None,
            "n_bins" :25
        }
    }
}

save         = True
figsize      = [6.4, 4.8]
dpi          = 200
rotate_ticks = False
scaling      = 'percent'
margin_shift_fraction = {
    'left'  : 0.20,
    'right' : 0.05,
    'top'   : 0.15,
    'bottom': 0.15
}
margin_axis_multiplier = -0.5
fontsize = 20
    
def plot_histogram(
        data,
        labels,
        xlabels,
        xlim=None,
        ylim=None,
        scaling='percent',
        n_bins=40,
        fontsize=20,
        figsize=[6.4, 4.8],
        rotate_ticks=False,
        margin_shift_fraction={
            'left'  : 0.20,
            'right' : 0.05,
            'top'   : 0.15,
            'bottom': 0.15
        },
        margin_axis_multiplier=-0.5
    ):
        
    scaling_factor_map = {
        'probability': 1,
        'percent'    : 100
    }
    scaling_name_map = {
        'probability': 'Fraction',
        'percent'    : 'Fraction [%]'
    }

    # set up subplots (1 x n_titles)
    n_components    = len(xlabels)
    figsize_current = (figsize[0]*n_components, figsize[1]  )
    fig, axes       = plt.subplots(
        ncols   =  n_components  ,
        sharex  =  "row"         ,
        figsize = figsize_current
    )
    axes = np.array(axes).reshape((n_components,))
    
    for col, (axis, data_axis, xlabel) in enumerate(zip(axes, data, xlabels)):

        # set x limits
        if xlim is None: 
            xlim = [None, None]
        if xlim[0] is None:
            xlim[0] = min([ data_file.min() for data_file in data_axis ])
        if xlim[1] is None:
            xlim[1] = max([ data_file.max() for data_file in data_axis ])

        # weights are used to normalize histogram
        weights_axis = [ np.ones_like(data_file)/len(data_file)*scaling_factor_map[scaling.lower()] for data_file in data_axis ]
        bins_axis    = np.linspace(xlim[0], xlim[1], n_bins+1)

        axis.hist(
            x=data_axis,
            bins=bins_axis,
            weights=weights_axis,
            edgecolor='black',
            linewidth=1.2
        )

        # set a common graph properties
        if col == 0:
            axis.get_figure().legend(
                labels         = labels        ,
                ncol           = len(data_axis),
                fontsize       = fontsize      ,
                framealpha     = 0.0           ,
                loc            = 'upper center',
                bbox_to_anchor = (0.5, 1.0)
            )

        # set only one y-label on the figure
        axis.set_ylabel('', fontsize=0)
        if col == 0:
            axis.set_ylabel(scaling_name_map[scaling.lower()], fontsize=fontsize)
        
        # set individual graph properties
        axis.set_xlabel(xlabel, fontsize=fontsize)
        axis.set_xlim  (xlim)
        if not ylim is None: axis.set_ylim(ylim)
        axis.tick_params(axis='both', which='major', labelsize=fontsize)
        axis.grid(visible=True, linestyle=':', linewidth=1)

        # rotate ticks
        if rotate_ticks:
            for label in axis.get_yticklabels():
                label.set_rotation(45)
            for label in axis.get_xticklabels():
                label.set_rotation(45)

    # adjust the margins
    # each subplot after the first lowers the amount of margin
    # due to each subsequent sublot not having a y-label
    margin_shift_multiplier = 1 + (n_components - 1) * margin_axis_multiplier
    fig.subplots_adjust(
        left   = 0 + margin_shift_fraction['left'  ] * margin_shift_multiplier,
        right  = 1 - margin_shift_fraction['right' ] * margin_shift_multiplier,
        bottom = 0 + margin_shift_fraction['bottom']                          ,
        top    = 1 - margin_shift_fraction['top'   ]
    )

    return fig, axes

def compare_statistics(inputs, save, figsize, dpi, rotate_ticks, scaling, fontsize, margin_shift_fraction, margin_axis_multiplier):

    for statistic, properties in zip(inputs['statistics'], inputs['statistics'].values()):
    
        # set up subplots (1 x n_titles)
        dims = (1, len(properties['xlabels']))
        figsize_current = [figsize[0]*len(properties['xlabels']), figsize[1]]
        fig, axes = plt.subplots(nrows=dims[0],ncols=dims[1],sharex="row",figsize=figsize_current)
        axes = np.array(axes).reshape(dims)
        
        # create the data array
        data = []
        for source in properties['paths']:

            path_file = properties['paths'][source]['file']
            path_hdf5 = properties['paths'][source]['hdf5']

            # import the data
            with h5py.File(path_file, 'r') as f:
                # [1:] removes error data
                file_data = f[path_hdf5][1:]
            # reshape adds component dimensions if necessary
            file_data = file_data.reshape((file_data.shape[0],-1))
            # cut off data there are no labels for
            if file_data.shape[1] > len(properties['xlabels']):
                print(f"WARNNG! more components than xlabels for {source}/{statistic}. Graphing only first {len(properties['xlabels'])} component(s)")
            file_data = file_data[:,:len(properties['xlabels'])]

            # initialize the data array if this is the first file
            # otherwise append additional components as needed
            if len(data) == 0:
                data = [ [component] for component in file_data.T ]
            else:
                for component_number in range(file_data.shape[1]):
                    data[component_number].append(file_data[:,component_number])

            
        plot_histogram(
            data                   = data                 , 
            labels                 = properties['paths']  ,
            xlabels                = properties['xlabels'],
            xlim                   = properties['xlim']   ,
            ylim                   = properties['ylim']   ,
            scaling                = scaling              ,
            n_bins                 = properties['n_bins'] ,
            fontsize               = fontsize             ,
            figsize                = figsize              ,
            rotate_ticks           = rotate_ticks         ,
            margin_shift_fraction  = margin_shift_fraction,
            margin_axis_multiplier = margin_axis_multiplier
        )

        if save:
            out_path_base, file_type = output_path.rsplit('.',1)
            plt.savefig(f"{out_path_base}_{statistic}.{file_type}", dpi=dpi)
        else:
            plt.show()
        fig.clf

compare_statistics(data, save, figsize, dpi, rotate_ticks, scaling, fontsize, margin_shift_fraction, margin_axis_multiplier)
