import tifffile, h5py, os, sys
import numpy as np
new_path = os.path.dirname(__file__)
if new_path not in sys.path:
    sys.path.append(new_path)
import utils_dream3d, utils_json, utils_python


def import_tiff_file(path_tiff):
    numbers_input = tifffile.imread(path_tiff)
    return numbers_input.shape, numbers_input

def import_text_file(path_text, delineator="\t"):
    with open(path_text) as file_input:
        shape, string_input = file_input.read().split("\n", 1)
    shape         = extract_numbers(shape        ,delineator=delineator, value_type=int).tolist()[0]
    numbers_input = extract_numbers(string_input,delineator=delineator)
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
    
#returns a slew of information based on how each orientation type should be handled
def get_properties():
    return utils_json.pull_input(os.path.realpath(new_path+"/"+"properties.json"))
    
def get_geometry(size_voxels, shape):
    dims = list(shape)[:-1][::-1]
    geometry = {
            "dims"       :dims,
            "origin"     :[0,0,0],
            "size_voxels":size_voxels
            }
    return geometry

def get_dream3d_data(column_types, numbers_input, geometry, properties, input_type, data={}):
    
    i = 0
    for column_type in column_types:
    
        # get properties
        if   column_type in properties.keys():
            n_components     = properties[column_type]["n_components"]
            datatype_python  = properties[column_type]["datatype_python"]
            datatype_dream3d = properties[column_type]["datatype_dream3d"]
            scale            = properties[column_type]["scale"]
            shift            = properties[column_type]["shift"]
        else:
            continue
        
        if input_type == "text":
            # obtain data pertaining to attribute array
            array_input = numbers_input[:,i:i+n_components]
            
            # reshape data to correct dimensions
            shape = tuple(geometry["dims"][::-1]+[n_components])
            array_input = array_input.reshape(shape)
            
        else:
            array_input = numbers_input[:, :, :, i:i+n_components]
        
        # scale and shift
        array_input = array_input*scale+shift
        
        # correct datatype
        array_input = array_input.astype(utils_dream3d.get_datatype(datatype_python))
        
        data[column_type] = {
            "data"                :array_input,
            "data_type"           :datatype_python,
            "attribute_array_type":datatype_dream3d
            }
        
        i += n_components

    return data
    
def append_dream3d_data(data, geometry, properties):

    # gives the sum of all of the arrays so that a threshold can be picked for "error" data
    data_new = {}
    for name, values in zip(data.keys(), data.values()):
        if values["data"].shape[-1] > 1:
        
            properties_new = {name+"Sum":{
                "n_components"    : 1,
                "datatype_python" : properties[name.replace("Sum","")]["datatype_python"],
                "datatype_dream3d": properties[name.replace("Sum","")]["datatype_dream3d"],
                "scale"           : 1,
                "shift"           : 0
            }}
        
            value = get_sum(values["data"])
        
            data_new.update(get_dream3d_data(
                [name+"Sum"], 
                value, 
                geometry, 
                properties_new, 
                "colour", 
                data=data_new
            ))
            
    data.update(data_new)
        
    # assume single phase if phase not passed 
    if not "Phases" in data.keys():
        
        name  = "Phases"
        value = np.ones((np.prod(geometry["dims"]), 1))*properties[name]["scale"]+properties[name]["shift"]
        
        data = get_dream3d_data(
            [name], 
            value, 
            geometry, 
            properties, 
            "text", 
            data=data
        )
        
    return data
    
#create a sum of the metric to spot low points
def get_sum(data):
    return np.sum(np.absolute(data),axis=3).reshape(np.append(data.shape[:-1],1).astype(tuple))

def export_data_hdf5(data, geometry, path_output, path_dream3d_input, path_VolumeDataContainer, path_crystallography):

    # specify hdf paths
    path_CellData         = path_VolumeDataContainer+"/"+"CellData"
    path_CellEnsembleData = path_VolumeDataContainer+"/"+"CellEnsembleData"
    path_Geometry         = path_VolumeDataContainer+"/"+"_SIMPL_GEOMETRY"

    with h5py.File(path_output, 'w') as file_output:
    
        # create hdf file and required data structure
        file_output.attrs["DREAM3D Version"] = utils_dream3d.format_string("1.2.815.6bed39e95A")
        file_output.attrs["FileVersion"]     = utils_dream3d.format_string("7.0")
        CellData = file_output.create_group(path_CellData)
        CellData.attrs["AttributeMatrixType"] = np.uint32([len(geometry["dims"])])
        CellData.attrs["TupleDimensions"]     = np.uint64(geometry["dims"])
        file_output.create_group("DataContainerBundles")
        Pipeline = file_output.create_group("Pipeline")
        Pipeline.attrs["Current Pipeline"] = utils_dream3d.format_string("Pipeline")
        Pipeline.attrs["Pipeline Version"] = np.int32([2])
        s = utils_dream3d.format_string(" \
        {\n \
            \"PipelineBuilder\":{\n \
                \"Name\":\"Pipeline\",\n \
                \"Number_Filters\":0,\n \
                \"Version\":6\n \
        } \
        ")
        Pipeline.create_dataset("Pipeline", data=s)
            
        # export cell ensemble data necessary for crystallographic analysis
        with h5py.File(path_dream3d_input, 'r') as file_dream3d_input:
            file_output.copy(file_dream3d_input[path_crystallography], path_CellEnsembleData)

        # export geometry data necessary for preserving dimensionality
        Geometry = file_output.create_group(path_Geometry)
        Geometry.attrs["GeometryName"]          = utils_dream3d.format_string("ImageGeometry")
        Geometry.attrs["GeometryType"]          = np.uint32([0])
        Geometry.attrs["GeometryTypeName"]      = utils_dream3d.format_string("ImageGeometry")
        Geometry.attrs["SpatialDimensionality"] = np.uint32([len(geometry["dims"])])
        Geometry.attrs["UnitDimensionality"]    = np.uint32([len(geometry["dims"])])
        Geometry.create_dataset("DIMENSIONS", data=geometry["dims"])
        Geometry.create_dataset("ORIGIN"    , data=geometry["origin"])
        Geometry.create_dataset("SPACING"   , data=geometry["size_voxels"])
        
        #export data to hdf
        for name, properties in zip(data.keys(), data.values()):
            utils_dream3d.insert_attribute_array(
                properties["data"],
                CellData,
                name,
                properties["attribute_array_type"]
                )
    
def push_attribute_arrays_expected(path_json, path_VolumeDataContainer, data, geometry):
    
    # pull dream3d pipeline
    data_json = utils_json.pull_input(path_json)
    
    # find the desired attribute matrix
    for i, data_container in enumerate(data_json["00"]["InputFileDataContainerArrayProxy"]["Data Containers"]):
        if data_container["Name"] == path_VolumeDataContainer.rsplit("/",1)[-1]:
            for j, attribute_matrix in enumerate(data_json["00"]["InputFileDataContainerArrayProxy"]["Data Containers"][i]["Attribute Matricies"]):
                if attribute_matrix["Name"] == "CellData":
                
                    # construct desired replacement data array
                    data_arrays = []
                    for name, properties in zip(data.keys(), data.values()):
                        data_arrays += [{
                            "Component Dimensions":[properties["data"].shape[-1]],
                            "Flag": 2,
                            "Name": name,
                            "Object Type": properties["attribute_array_type"],
                            "Path": path_VolumeDataContainer+"/"+"CellData",
                            "Tuple Dimensions": geometry["dims"],
                            "Version": 2
                        }]
                    data_json["00"]["InputFileDataContainerArrayProxy"]["Data Containers"][i]["Attribute Matricies"][j]["Data Arrays"] = data_arrays
                    
                    # push inputs to dream3d pipeline
                    utils_json.push_input(path_json, data_json, padding=4*" ")
                    return
   
def convert_slicegan_to_dream3d(dir_path, path_project, name_project, plane_suffixes, dream3d_path, orientations_types, input_type):

    ##set paths
    #directory paths
    path_json                 = os.path.realpath(dir_path+"/dream3d_support/"+"1-slicegan_data-"+orientations_types[0]+".json")
    path_dream3d_input        = os.path.realpath(dir_path+"/dream3d_support/pipeline_output/"+"0-ebsd_data.dream3d")
    path_slice_image_in       = os.path.realpath(dir_path+"/dream3d_support/pipeline_output/"+"slices.png")
    path_slicegan_input_text  = os.path.realpath(path_project+"/"+name_project+"/"+name_project+".txt")
    path_slicegan_input_tiff  = os.path.realpath(path_project+"/"+name_project+"/"+name_project+".tif")
    path_output               = os.path.realpath(path_project+"/"+name_project+"/"+"1-slicegan_data.dream3d")
    path_dream3d_output       = os.path.realpath(path_project+"/"+name_project+"/"+"2-slicegan_data.dream3d")
    path_slice_image_out      = os.path.realpath(path_project+"/"+name_project)
    #hdf5 path
    path_VolumeDataContainer  = "/DataContainers/ImageDataContainer"
    
    print("Converting SliceGAN files to DREAM.3D files...")
    # import data
    if input_type == "text":
        shape, numbers_input = import_text_file(path_slicegan_input_text)
    else:
        shape, numbers_input = import_tiff_file(path_slicegan_input_tiff)
    #import ebsd reference data
    name_planes, size_voxels, path_crystallography = \
        utils_dream3d.import_data_ebsd_reference(path_dream3d_input, path_VolumeDataContainer, orientations_types)
    #get information pertaining to orientation types
    properties = get_properties()
    #get geometry information (dims, origin, resolution)
    geometry = get_geometry(size_voxels, shape)
    #import slicegan data (as a list, grouped by attribute array)
    data = get_dream3d_data(orientations_types, numbers_input, geometry, properties, input_type)
    #append extra data as needed
    data = append_dream3d_data(data, geometry, properties)
    #export to hdf5
    export_data_hdf5(data, geometry, path_output, path_dream3d_input, path_VolumeDataContainer, path_crystallography)
    print("Done Converting")
    
    if os.path.exists(path_json):
    
        print("Anchoring DREAM.3D paths...")
        # anchor dream3d json files
        utils_dream3d.replace_json_paths(path_json,plane_suffixes,input_path=path_output,output_path=path_dream3d_output,image_path=path_slice_image_in)
        print("Done Anchoring")
        
        push_attribute_arrays_expected(path_json, path_VolumeDataContainer, data, geometry)
    
        print("Constructing .XDMF and .PNG files...")
        #run dream.3d file (to convert orientations to useful values, find "error" data, create xdmf files, etc...)
        utils_dream3d.call_dream3d(dream3d_path, path_json)
        #copy over dream3d image
        utils_python.data_copy(path_slice_image_in, path_slice_image_out)
        print("Done Constructing")
    