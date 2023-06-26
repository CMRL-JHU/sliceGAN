import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


output_path = "pipeline_output/07-compare_statistics.pdf"

save         = True
fontsize     = 20
figsize      = [6.4, 4.8]
dpi          = 200
rotate_ticks = False
margin_shift_fraction = {
    'left'  : 0.20,
    'right' : 0.05,
    'top'   : 0.15,
    'bottom': 0.15
}
margin_axis_multiplier = -0.5
stat         = 'percent'

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
    
def plot_histogram(
        data,
        xlim=None,
        ylim=None,
        n_bins=40,
        stat='percent',
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

    map_stat = {
        'percent'    : 'Fraction [%]', # heights sum to 100
        'probability': 'Fraction'    , # heights sum to 1
        'density'    : 'Density'     , # areas sum to 1
        'count'      : 'Count'       , # count
        'frequency'  : 'Frequency'     # count/bin_width
    }

    # set up subplots (1 x n_titles)
    n_components    = len(data.name_component.unique())
    figsize_current = (figsize[0]*n_components, figsize[1]  )
    fig, axes       = plt.subplots(
        ncols   =  n_components  ,
        sharex  =  "row"         ,
        figsize = figsize_current
    )
    axes = np.array(axes).reshape((n_components,))
    
    for col, (axis, name_component) in enumerate(zip(axes, data.name_component.unique())):
        
        data_component = data[data.name_component==name_component]

        # set x limits
        if xlim is None: 
            xlim = [None, None]
        if xlim[0] is None:
            xlim[0] = data_component.value.min()
        if xlim[1] is None:
            xlim[1] = data_component.value.max()

        # create bins
        bins    = np.linspace(xlim[0], xlim[1], n_bins+1)

        sns.histplot(
            data        = data_component,
            x           = "value"       ,
            stat        = stat          ,
            hue         = 'name_source' ,
            bins        = bins          ,
            multiple    = 'dodge'       ,
            common_norm = False         ,
            shrink      = 0.8           ,
            ax          = axis          ,
            edgecolor   = 'k'           ,
            linewidth   = 0
        )

        # set only one legend on the figure
        axis.get_legend().set_visible(False)
        if col == 0:
            # handles, labels = axis.get_legend_handles_labels()
            handles = axis.get_legend().legendHandles
            labels  = [ text._text for text in axis.get_legend().texts]
            axis.get_figure().legend(handles, labels)
            sns.move_legend(
                obj            = axis.get_figure()             ,
                loc            = "upper center"                ,
                bbox_to_anchor = (0.5, 1.0)                    ,
                ncol           = len(data.name_source.unique()),
                title          = None                          ,
                frameon        = False                         ,
                fontsize       = 20
            )

        # set only one y-label on the figure
        axis.set_ylabel('', fontsize=0)
        if col == 0:
            axis.set_ylabel(map_stat[stat], fontsize=fontsize)
        
        axis.set_xlabel(name_component, fontsize=fontsize)
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

def compare_statistics(inputs, save, fontsize, figsize, dpi, rotate_ticks, stat, margin_shift_fraction, margin_axis_multiplier):

    for statistic, properties in zip(inputs['statistics'], inputs['statistics'].values()):
    
        # initialize the data array
        data = pd.DataFrame()

        for source in properties['paths']:

            # import the data
            with h5py.File(properties['paths'][source]['file'], 'r') as f:
                # [1:] removes error data
                file_data = f[properties['paths'][source]['hdf5']][1:]
            # reshape adds component dimensions if necessary
            file_data = file_data.reshape((file_data.shape[0],-1))

            # warn user that data there are no labels for will be cut off
            if file_data.shape[1] > len(properties['xlabels']):
                print(f"WARNNG! more components than xlabels for {source}/{statistic}. Graphing only first {len(properties['xlabels'])} component(s)")

            for component_data, component_name in zip(file_data.T, properties['xlabels']):

                # fill the data array
                data = pd.concat(
                    [
                        data,
                        pd.DataFrame({
                            'value'         : component_data,
                            'name_component': component_name,
                            'name_source'   : source
                        })
                    ],
                    ignore_index=True
                )

        fig, axes = plot_histogram(
            data                   = data                  ,
            xlim                   = properties['xlim']    ,
            ylim                   = properties['ylim']    ,
            n_bins                 = properties['n_bins']  ,
            stat                   = stat                  ,
            fontsize               = fontsize              ,
            figsize                = figsize               ,
            rotate_ticks           = rotate_ticks          ,
            margin_shift_fraction  = margin_shift_fraction ,
            margin_axis_multiplier = margin_axis_multiplier
        )

        if save:
            out_path_base, file_type = output_path.rsplit('.',1)
            plt.savefig(f"{out_path_base}_{statistic}.{file_type}", dpi=dpi)
        else:
            plt.show()
        fig.clf()

compare_statistics(data, save, fontsize, figsize, dpi, rotate_ticks, stat, margin_shift_fraction, margin_axis_multiplier)
