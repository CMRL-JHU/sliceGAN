import h5py
import argparse
import utils_dream3d

path_input            = "./pipeline_output/4-watershed.dream3d"
path_output           = "./pipeline_output/5-watershed.dream3d"
path_Geometry         = "/DataContainers/ImageDataContainer/_SIMPL_GEOMETRY"
path_CellData         = "/DataContainers/ImageDataContainer/CellData"
path_CellEnsembleData = "/DataContainers/ImageDataContainer/CellEnsembleData"

whitelist = [
    "EulerAngles"                    ,
    "Phases"                         ,
    "Error_Mask"                     ,
    "WatershedBasinMean"             ,
    "WatershedBasinStandardDeviation",
    "WatershedBasins"
]

with h5py.File(path_input , 'r') as file_input:
    
    geometry = {
        "dims"       : file_input[path_Geometry+"/"+"DIMENSIONS"],
        "origin"     : file_input[path_Geometry+"/"+"ORIGIN"    ],
        "size_voxels": file_input[path_Geometry+"/"+"SPACING"   ]
    }
    
    utils_dream3d.create_file_dream3d(
        path_output
    )
    utils_dream3d.insert_geometry(
        path_output,
        geometry,
        path_Geometry
    )
    utils_dream3d.copy_crystallography(
        path_output,
        path_input,
        path_CellEnsembleData,
        path_CellEnsembleData
    )
    for name in list(set([*file_input[path_CellData]]).intersection(set(whitelist))):
        
        # hacky work-around for matlab strings
        try:
            dtype = file_input[path_CellData+"/"+name].attrs["ObjectType"]   .decode('UTF-8')
        except:
            dtype = file_input[path_CellData+"/"+name].attrs["ObjectType"][0].decode('UTF-8')
            
        utils_dream3d.insert_attribute_array(
            path_output                                   ,
            path_CellData                                 ,
            name                                          ,
            data = file_input[path_CellData+"/"+name][...],
            dtype = dtype                                 ,
            attribute_matrix_type = "Cell"
        )
        
utils_dream3d.make_xdmf(path_output)
