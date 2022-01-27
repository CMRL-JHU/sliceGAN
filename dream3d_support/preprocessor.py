import tifffile, h5py, os, sys
import numpy as np
new_path = os.path.dirname(__file__)
if new_path not in sys.path:
    sys.path.append(new_path)
import utils_dream3d, utils_json, plot_hist

def anchor_dream3d_paths(plane_suffixes, ebsd_paths, output_path, json_path):
    utils_dream3d.replace_json_paths(json_path, plane_suffixes, ebsd_paths=ebsd_paths, output_path=output_path)
    
def convert_ebsd_to_dream3d(dream3d_path, json_path):
    utils_dream3d.call_dream3d(dream3d_path, json_path)

def rotate_planes(path_dream3d_input, plane_suffixes, name_planes, plane_rotations):
    
    with h5py.File(path_dream3d_input, 'r+') as file_input:
        
        # collect all the data containers
        DataContainers = [file_input["DataContainers"+"/"+DataContainer] for DataContainer in file_input["DataContainers"]]

        # rotate_planes
        for DataContainer, name_plane in zip(DataContainers, name_planes):
            rotate_plane(plane_suffixes,name_plane,DataContainer,plane_rotations)
            
def rotate_plane(plane_suffixes,name,DataContainer,plane_rotations):
    for i in range(len(plane_suffixes)):
        if name == plane_suffixes[i]:
            utils_dream3d.rotate_all(DataContainer,plane_rotations[i])

#convert from dream3d file to tif images
def convert_dream3d_to_tiff(path_dream3d_input, path_output, orientations_types, orientations_names, name_planes, data_type, plot_orientations=False):

    with h5py.File(path_dream3d_input, 'r') as file_input:
    
        # collect all the data containers
        DataContainers = [file_input["DataContainers"+"/"+DataContainer] for DataContainer in file_input["DataContainers"]]
            
        #export image data to tif file
        for DataContainer, name_plane in zip(DataContainers, name_planes):
            
            
            data = get_data(orientations_types,orientations_names,DataContainer)
            if data_type == "text":
                path = os.path.realpath(path_output+name_plane+".txt")
                export_text_file(path, data)
            else:
                path = os.path.realpath(path_output+name_plane+".tif")
                export_tiff_file(path, data)
            
def get_data(orientations_types,orientations_names,DataContainer):
    
    properties = get_properties()
    
    for i, (orientations_type, orientations_name) in enumerate(zip(orientations_types, orientations_names)):
    
        # get properties
        scale = properties[orientations_type]["scale"]
        shift = properties[orientations_type]["shift"]
    
        # remove empty Z container and scale data to [0,1]
        data = utils_dream3d.remove_empty_z_container(DataContainer["CellData/"+orientations_name][:]-shift)/scale
    
        # concatenate all the orientation_types together along the last axis
        if i == 0:
            data_concatenated = data
        else:
            # data should be of shape [y, x, component], and we want to append the data as an extra component
            data_concatenated = np.append(data_concatenated, data, axis=2)

    return data_concatenated

def get_properties():
    return utils_json.pull_input(os.path.realpath(new_path+"/"+"properties.json"))
    
def export_text_file(path_output, data, number_format='%.0f', delimiter="\t"):

    dims         = list(data.shape[:-1][::-1])
    n_components = data.shape[-1 ]
        
    # break apart the list into components
    shape = (np.prod(dims).astype(int), n_components)
    data = data.reshape(shape)
    
    
    with open(path_output, "w") as file_output:
        file_output.write(delimiter.join(map(str, dims[::-1]+[n_components]))+"\n")
    with open(path_output, "ab") as file_output:
        np.savetxt(file_output, data, fmt=number_format, delimiter=delimiter)
    
def export_tiff_file(path, data):
    tifffile.imwrite(path,data)

def convert_ebsd_to_tiff(dir_path,project_name,ebsd_paths,path_dream3d,path_output,plane_suffixes,plane_rotations,output_type,orientations_types,orientations_names,n_dims=3,plot_orientations=False):
    
    # directory paths
    path_import_pipeline = os.path.realpath(dir_path+"/dream3d_support/"+f"0-ebsd_data.{n_dims}d.{ebsd_paths[0].rsplit('.',1)[1]}.json")
    path_dream3d_input = os.path.realpath(dir_path+"/dream3d_support/pipeline_output/0-ebsd_data.dream3d")
    # ebsd_paths = [os.path.realpath(dir_path+"/Input/"+project_name+"/"+ebsd_path) for ebsd_path in ebsd_paths] ########### doesn't work with optimize_parameters
    ebsd_paths = [os.path.realpath(dir_path+"/Input/"+ebsd_path) for ebsd_path in ebsd_paths]
    # hdf5 path
    path_VolumeDataContainer = "/DataContainers/ImageDataContainer"
    
    ##################temp ##########need to pass all the actual values we should read in here
    # orientations_types = [orientations_type]
    # orientations_names = [orientations_name]
    
    
    print("Anchoring DREAM.3D paths...")
    anchor_dream3d_paths(plane_suffixes, ebsd_paths, path_dream3d_input, path_import_pipeline)
    print("Done Anchoring")
    
    print("Converting EBSD files to DREAM.3D files...")
    convert_ebsd_to_dream3d(path_dream3d, path_import_pipeline)
    print("Done Converting")
    
    print("Converting DREAM.3D files to SliceGAN files...")
    # required only for error checking and name_planes
    _, _, name_planes = utils_dream3d.import_data_ebsd_reference(path_dream3d_input, path_VolumeDataContainer, orientations_types)
    # rotate planes to correct orientations for slicegan (see references)
    rotate_planes(path_dream3d_input, plane_suffixes, name_planes, plane_rotations)
    # convert .dream3d files to either .tif or .txt files
    convert_dream3d_to_tiff(path_dream3d_input, path_output, orientations_types, orientations_names, plane_suffixes, output_type, plot_orientations)
    print("Done converting")