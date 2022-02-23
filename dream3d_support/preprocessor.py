import tifffile, h5py, os, sys
import numpy as np
new_path = os.path.dirname(__file__)
if new_path not in sys.path:
    sys.path.append(new_path)
import utils_dream3d, utils_json, plot_hist

def anchor_dream3d_paths(json_path, input_path, output_path):
    utils_dream3d.replace_json_paths(json_path,input_path=input_path,output_path=output_path)
    utils_dream3d.update_attribute_arrays_expected(json_path)
    
def convert_ebsd_to_dream3d(path_dir, ebsd_paths, path_dream3d):
    
    ebsd_paths_new = []
    for path_ebsd in ebsd_paths:

        name, type_input = path_ebsd.rsplit('.',1)
        type_input = type_input.lower()

        if type_input == "dream3d":
            ebsd_paths_new += [path_ebsd]
            continue

        path_json   = os.path.realpath(path_dir+"/dream3d_support/"+"0-ebsd_data."+type_input+".json")
        name_output = name+".dream3d"
        ebsd_paths_new += [name_output]

        anchor_dream3d_paths(path_json, path_ebsd, name_output)
        utils_dream3d.call_dream3d(path_dream3d, path_json)

    return ebsd_paths_new

def rotate_planes(ebsd_paths, plane_rotations, path_CellData, path_Geometry):
    
    for ebsd_path, plane_rotation in zip(ebsd_paths, plane_rotations):
        utils_dream3d.rotate_all(ebsd_path, plane_rotation, path_CellData, path_Geometry)   

#convert from dream3d file to tif images
def convert_dream3d_to_tiff(ebsd_paths, path_output, path_CellData, plane_names, orientations_types, orientations_names, data_type, plot_orientations=False):

    for ebsd_path, plane_name in zip(ebsd_paths, plane_names):
    
        data = get_data(orientations_types,orientations_names,ebsd_path,path_CellData)

        if data_type == "text":
            path = os.path.realpath(path_output+"_"+plane_name+".txt")
            export_text_file(path, data)
        else:
            path = os.path.realpath(path_output+"_"+plane_name+".tif")
            export_tiff_file(path, data)
            
def get_data(orientations_types,orientations_names,ebsd_path,path_CellData):
    
    properties = get_properties()
    
    for i, (orientations_type, orientations_name) in enumerate(zip(orientations_types, orientations_names)):
    
        # get properties
        scale = properties[orientations_type]["scale"]
        shift = properties[orientations_type]["shift"]
    
        # import from hdf5
        with h5py.File(ebsd_path, 'r') as file_input:
            data = file_input[path_CellData+"/"+orientations_name][:]
        # remove empty Z container
        data = utils_dream3d.remove_empty_z_container(data)
        # scale data to [0,1]
        data = (data-shift)/scale
    
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
    
    # write out dimensions
    with open(path_output, "w") as file_output:
        file_output.write(delimiter.join(map(str, dims[::-1]+[n_components]))+"\n")

    # write out data
    with open(path_output, "ab") as file_output:
        np.savetxt(file_output, data, fmt=number_format, delimiter=delimiter)
    
def export_tiff_file(path, data):
    tifffile.imwrite(path,data)

def convert_ebsd_to_tiff(dir_path,project_name,ebsd_paths,plane_names,path_dream3d,path_output,plane_rotations,data_type,orientations_types,orientations_names,plot_orientations=False):

    # directory paths
    # ebsd_paths = [os.path.realpath(dir_path+"/Input/"+project_name+"/"+ebsd_path) for ebsd_path in ebsd_paths] ########### doesn't work with optimize_parameters
    ebsd_paths = [os.path.realpath(dir_path+"/Input/"+ebsd_path) for ebsd_path in ebsd_paths]
    # hdf5 paths
    path_VolumeDataContainer = "/DataContainers/ImageDataContainer"
    path_Geometry = path_VolumeDataContainer+"/"+"_SIMPL_GEOMETRY"
    path_CellEnsembleData = path_VolumeDataContainer+"/"+"CellEnsembleData"
    path_CellData = path_VolumeDataContainer+"/"+"CellData"

    # convert to dream3d
    print("Converting EBSD files to DREAM.3D files...")
    ebsd_paths = convert_ebsd_to_dream3d(dir_path, ebsd_paths, path_dream3d)
    print("Done converting")

    # A microstructure with no common resolutions or with differing crystallographic data
    # will not be able to be merged together at the end of the training run.
    # To avoid wasting training time, these merge errors are caught before the run begins
    print("Checking for conflicts in resolution and crystallography...")
    resolution = utils_dream3d.import_resolution(ebsd_paths, plane_names, path_Geometry)
    utils_dream3d.check_error_crystallography(ebsd_paths, path_CellEnsembleData, orientations_types)
    print("No conflicts found")

    # rotate planes to correct orientations for slicegan (see references)
    print("Rotating planes...")
    rotate_planes(ebsd_paths, plane_rotations, path_CellData, path_Geometry)
    print("Done rotating")

    # convert .dream3d files to either .tif or .txt files
    print("Converting DREAM.3D files to SliceGAN files...")
    convert_dream3d_to_tiff(ebsd_paths, path_output, path_CellData, plane_names, orientations_types, orientations_names, data_type, plot_orientations)
    print("Done converting")
