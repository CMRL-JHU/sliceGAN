import h5py, json, os, sys, subprocess, platform, math
import numpy as np

#returns numpy class of datatype
def get_datatype(datatype):
    if datatype == "float32":
        return np.float32
    if datatype == "int32":
        return np.int32
    if datatype == "uint8":
        return np.uint8
        
# Find all filter ids in a DREAM.3D *.JSON file with name
def get_filter_ids(data_json, name_filter):

    filter_ids = []
    for filter_id in data_json.keys():
    
        if "Filter_Name" in data_json[filter_id].keys() \
            and data_json[filter_id]["Filter_Enabled"]  \
            and data_json[filter_id]["Filter_Name"] == name_filter:
            
            filter_ids += [filter_id]
            
    return filter_ids
        
#slicegan has no knowlegde of crystallography data, resolution, or naming schema, so it is imported from the .dream3d file
def import_data_ebsd_reference(path_dream3d_input, VolumeDataContainer_Path, orientations_type):
    
    with h5py.File(path_dream3d_input, 'r') as file_dream3d_input:
    
        # Search through *.DREAM3D file to find all data containers
        DataContainers = []
        for DataContainer in file_dream3d_input["DataContainers"]:
            DataContainers.append(file_dream3d_input["DataContainers"+"/"+DataContainer])
            
        # Search through all data containers to find reference information
        name_planes       = []
        size_voxels       = []
        CellEnsembleData = []
        for DataContainer in DataContainers:
            # Because there are 3 images in the source file,
            # each data container contains a plane represented by two basis vectors trailing the path name:
            #     ('yz','xz', or 'xy')
            name_planes      += [filter_string(DataContainer.name, [VolumeDataContainer_Path], "xyzXYZ").lower()]
            # Each image is 2D, so the spacing in Z dimension is inconsequential
            size_voxels      += [DataContainer["_SIMPL_GEOMETRY/SPACING"][:-1].tolist()]
            CellEnsembleData.append(DataContainer["CellEnsembleData"])
        path_crystallography = CellEnsembleData[0].name
        
        # If the input contains 3 planes, we are trying to reconstruct a 3D microstructure.
        # The side resolutions should be the same in order to reconstruct properly.
        # This function allows us to determine if the side resolutions are compatible
        # And gives us the common resolutions [x,y,z] if they exist
        if len(name_planes) == 3:
            size_voxels = find_common_global_value(name_planes, size_voxels)
        # If the input contains 1 plane, we're trying to reconstruct a 2D microstructure.
        # No compatibility check is required, and the resolution is given as [x,y,1]
        else:
            size_voxels = size_voxels+[1]
        
        # A microstructure with no common resolutions or with differing crystallographic data
        # will not be able to be merged together at the end of the training run.
        # To avoid wasting training time, these merge errors are caught before the run begins
        check_errors(name_planes, size_voxels, CellEnsembleData, orientations_type)
        
    return name_planes, size_voxels, path_crystallography
    
#filter a string with blacklist and whitelist
def filter_string(string, blacklist=None, whitelist=None):
    if not blacklist is None:
        for entry in blacklist:
            string = string.replace(entry, "")
    if not whitelist is None:
        string = "".join(list(filter(lambda char: char in set(whitelist), string)))
    return string
    
#check for errors that would prevent data reconciliation
def check_errors(name_planes, size_voxels, CellEnsembleData, orientations_type):
    if len(name_planes) < 3 and not is_crystallographic(orientations_type):
        pass
    elif len(name_planes) < 3 and is_crystallographic(orientations_type):
        check_error_crystallography(CellEnsembleData,orientations_type)
    elif len(name_planes) == 3 and not is_crystallographic(orientations_type):
        check_error_resolution(size_voxels)
    else:
        check_error_crystallography(CellEnsembleData,orientations_type)
        check_error_resolution(size_voxels)
        
#check if there is a common voxel size for each side. if there is not, then there will be no way to reconcile the dream3d data with the slicegan data at the end.
def check_error_resolution(size_voxels):
    if isinstance(size_voxels, str):
        raise ValueError(size_voxels)
    
#check if the crystallography matches exactly. if it does not, then there will be no way to reconcile the dream3d data with the slicegan data at the end.
def check_error_crystallography(CellEnsembleData, orientations_type):
    if is_crystallographic(orientations_type):
        if not groups_are_equal(CellEnsembleData):
            raise RuntimeError("Crystallography data does not match between all 3 planes. No way to link crystallography")
            
#returns true if the value is a crystallographic data type
def is_crystallographic(orientations_types):
    if type(orientations_types) in [list, tuple]:
        result = []
        for orientations_type in orientations_types:
            result += [is_crystallographic(orientations_type)]
        return any(result)
    return orientations_types=="Quats" or orientations_types=="EulerAngles"
    
#EBSD scans contain two directional dimensions, but dream3d fills in the third to voxelize them
#dream3d seems to always keep the images oriented in the xy plane, so the z axis is filled with 1
#dream3d also seems to organize its directional dimensions backwards (z,y,x), so we remove dim 0
def remove_empty_z_container(array):
    return array.reshape(tuple(list(array.shape)[1:]))
    
#Replace the empty z container to put arrays back into dream3d
def replace_empty_z_container(array):
    return array.reshape(tuple([1]+list(array.shape)))
    
#rotate the 2-D array 90 degrees
#dream 3d keeps images oriented in a local xy plane, so to rotate, just swap the xy axes
#then flip either the x or y data to rotate counter-clockwise or clockwise respectively
def rotate_90_degrees(array, direction=1, n=1):
    for i in range(n):
        if direction == -1:
            array = array.swapaxes(0,1)[:,::-1,:] 
        else:
            array = array.swapaxes(0,1)[::-1,:,:]
    return array
    
#rotate all the CellData arrays within dream3d
def rotate_all(DataContainer,n):
    for i in range(n):
        dims = DataContainer["CellData"].attrs["TupleDimensions"]
        dims[0], dims[1] = dims[1], dims[0]
        DataContainer["CellData"].attrs["TupleDimensions"] = np.uint64(dims)
        DataContainer["_SIMPL_GEOMETRY/DIMENSIONS"][:] = np.uint64(dims)
        size_voxels = DataContainer["_SIMPL_GEOMETRY/SPACING"][:]
        size_voxels[0], size_voxels[1] = size_voxels[1], size_voxels[0]
        DataContainer["_SIMPL_GEOMETRY/SPACING"][:] = size_voxels
        for item in DataContainer["CellData"]:
            DataContainer["CellData/"+item].attrs["TupleDimensions"] = np.uint64(dims)
            DataContainer["CellData/"+item].attrs["Tuple Axis Dimensions"] = format_string("x="+str(dims[0])+",y="+str(dims[1])+",z="+str(dims[2]))
            if   DataContainer["CellData/"+item].attrs["ObjectType"] == format_string("DataArray<float>"):
                DataContainer["CellData/"+item][:] = np.float32(replace_empty_z_container(rotate_90_degrees(remove_empty_z_container(DataContainer["CellData/"+item][:]), n=n)))
            elif DataContainer["CellData/"+item].attrs["ObjectType"] == format_string("DataArray<int32_t>"):
                DataContainer["CellData/"+item][:] = np.int32(replace_empty_z_container(rotate_90_degrees(remove_empty_z_container(DataContainer["CellData/"+item][:]), n=n)))

# Create an empty DREAM.3D file with just the bare minimum to be recognized by DREAM.3D
def create_file_dream3d(path_output):

    with h5py.File(path_output, 'w') as file_output:
    
        # create hdf file and required data structure
        file_output.attrs["DREAM3D Version"] = format_string("1.2.815.6bed39e95A")
        file_output.attrs["FileVersion"]     = format_string("7.0")
        file_output.create_group("DataContainerBundles")
        Pipeline = file_output.create_group("Pipeline")
        Pipeline.attrs["Current Pipeline"] = format_string("Pipeline")
        Pipeline.attrs["Pipeline Version"] = np.int32([2])
        s = format_string(" \
        {\n \
            \"PipelineBuilder\":{\n \
                \"Name\":\"Pipeline\",\n \
                \"Number_Filters\":0,\n \
                \"Version\":6\n \
        } \
        ")
        Pipeline.create_dataset("Pipeline", data=s)

# Geometry should be a dictionary formatted as:
# {
#    "dims"       : [x,y,z]
#    "origin"     : [x,y,z] # normally [0,0,0]
#    "size_voxels": [x,y,z]
# }
def insert_geometry(
    path_output,
    geometry,
    path_Geometry="/DataContainers/ImageDataContainer/_SIMPL_GEOMETRY"
    ):

    with h5py.File(path_output, 'a') as file_output:

        # export geometry data necessary for preserving dimensionality
        Geometry = file_output.create_group(path_Geometry)
        Geometry.attrs["GeometryName"]          = format_string("ImageGeometry")
        Geometry.attrs["GeometryType"]          = np.uint32([0])
        Geometry.attrs["GeometryTypeName"]      = format_string("ImageGeometry")
        Geometry.attrs["SpatialDimensionality"] = np.uint32([len(geometry["dims"])])
        Geometry.attrs["UnitDimensionality"]    = np.uint32([len(geometry["dims"])])
        Geometry.create_dataset("DIMENSIONS", data=geometry["dims"])
        Geometry.create_dataset("ORIGIN"    , data=geometry["origin"])
        Geometry.create_dataset("SPACING"   , data=geometry["size_voxels"])
        
# Crystallography information comes from EBSD files of many types
# While you could create your own EBSD data from an EBSD file (.ang, .ctf, etc...)
# DREAM.3D handles this pretty well already, so it's likely best to import it with DREAM.3D
# This function can be used to copy already formatted crystallography information from a reference *.DREAM3D file
def copy_crystallography(
    path_dream3d_output,
    path_dream3d_input,
    path_CellEnsembleData_output="/DataContainers/ImageDataContainer/CellEnsembleData",
    path_CellEnsembleData_input="/DataContainers/ImageDataContainer/CellEnsembleData"
    ):
    
    with h5py.File(path_dream3d_input, 'r') as file_dream3d_input, h5py.File(path_dream3d_output, 'a') as file_dream3d_output:
    
        # export cell ensemble data necessary for crystallographic analysis
        file_dream3d_output.copy(file_dream3d_input[path_CellEnsembleData_input], path_CellEnsembleData_output)

# Insert dream3d formatted attribute array with specified data into specified hdf5 group
# Should should be a numpy array of shape: [z,y,x,components] or [x,components]
# A missing component dimension can be inferred, but it's best to be explicit
def insert_attribute_array(
    path_output,
    name,
    data,
    dtype="DataArray<float>",
    path_CellData="/DataContainers/ImageDataContainer/CellData",
    path_Geometry="/DataContainers/ImageDataContainer/_SIMPL_GEOMETRY"
    ):

    ## Find dimension and component information
    # Assume the data is 2 or 3 dimensional
    if len(data.shape) > 2:
        # Assume the data is missing the component dimension
        if not len(data.shape) > 3:
            data = data.reshape(np.append(data.shape,1).astype(tuple))
    # Assume the data is 1 dimensional
    else:
        # Assume the data is missing the component dimension
        if not len(data.shape) > 1:
            data = data.reshape(np.append(data.shape,1).astype(tuple))
    dims                  = list(data.shape[:-1][::-1])
    components            = [data.shape[-1]]
    tuple_axis_dimensions = ','.join(['='.join([name,str(value)]) for name,value in zip(['x','y','z'],dims)])
        
    with h5py.File(path_output, 'a') as file_output:
        
        # create the CellData group and its required attributes if it does not exist
        group = file_output.require_group(path_CellData)
        attributes = {
            "AttributeMatrixType": np.uint32([len(file_output[path_Geometry+"/"+"DIMENSIONS"][...])]),
            "TupleDimensions"    : np.uint64(file_output[path_Geometry+"/"+"DIMENSIONS"][...])
        }
        for key, val in zip(attributes.keys(), attributes.values()):
            if not key in [i for i in file_output[path_CellData].attrs.keys()]:
                group.attrs[key] = val
        
        # export attribute array
        dataset = group.create_dataset(name, data=data)
        dataset.attrs["ComponentDimensions"]   = np.uint64(components)
        dataset.attrs["DataArrayVersion"]      = np.int32([2])
        dataset.attrs["ObjectType"]            = format_string(dtype)
        dataset.attrs["Tuple Axis Dimensions"] = format_string(tuple_axis_dimensions)
        dataset.attrs["TupleDimensions"]       = np.uint64(dims)

#call dream3d file
def call_dream3d(dream3d_path, json_path):
    def quote(string):
        return "\""+string+"\""
    def path(string):
        if platform.system() == "Windows":
            return quote(os.path.realpath(string))
        else:
            return os.path.realpath(string)
    if platform.system() == "Windows":
        subprocess.call(path(dream3d_path+"/PipelineRunner.exe")+" -p "+path(json_path))
    else:
        subprocess.call([path(dream3d_path+"/bin/PipelineRunner"),"-p",path(json_path)])

#replace the file paths in dream3d file because dream3d cannot do relative pathing
def replace_json_paths(json_path,plane_names,input_path=None,ebsd_paths=None,output_path=None,image_path=None):

    def pad_with_zeros(current_value, max_value):
        return (int(np.log10(max_value))-int(np.log10(current_value)))*"0"+str(current_value)
        
    def numeric_dict(data):
        return {key:value for key, value in zip(data.keys(), data.values()) if key.isnumeric()}

    def find_next_name_change(data,item):

        # filters always have numeric ids. only search through those
        data = numeric_dict(data)

        # filter ids after the filter of interest
        filter_ids = [pad_with_zeros(filter_id, len(data)) for filter_id in range(int(item)+1,len(data))]
        
        for filter_id in filter_ids:
        
            # catch ran into next data container reader
            if data[filter_id]["Filter_Name"] == "DataContainerReader":
                break
            
            #dream3d pipeline runner cannot read from groups that have been renamed. that was a fun bug hunt...
            if data[filter_id]["Filter_Name"] == "CopyDataContainer":
                return data[filter_id]["NewDataContainerName"]
                
        raise ValueError("No next name change")
    
    # Read the DREAM.3D *.JSON file
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        
    ## Replace input paths
    # For multiple input files
    if ebsd_paths is not None:
    
        for plane_name, ebsd_path in zip(plane_names, ebsd_paths):
        
            # For *.CTF data
            for filter_id in get_filter_ids(data, "ReadCtfData"):
                if plane_name.lower() in data[filter_id]["DataContainerName"].replace("ImageDataContainer","").lower():
                    data[filter_id]["InputFile"] = ebsd_path
                    
            # For *.DREAM3D data
            for filter_id in get_filter_ids(data, "DataContainerReader"):
                # DREAM.3D cannot assign a name to an imported data container upon load
                # The data container must be renamed,
                # BUT the pipeline runner cannot read renamed data containers
                # So the data container must be copied with a new name and then deleted
                # Therefor the new name must be found somewhere after the DataContainerReader
                name = find_next_name_change(data, filter_id)
                if plane_name.lower() in name.replace("ImageDataContainer","").lower():
                    data[filter_id]["InputFile"] = ebsd_path
    
    # For a single input file
    elif input_path is not None:
        filter_id = get_filter_ids(data, "DataContainerReader")[0]
        data[filter_id]["InputFile"] = input_path
            
    # Replace *.DREAM3D output paths
    if output_path is not None:
        filter_id = get_filter_ids(data, "DataContainerWriter")[0]
        data[filter_id]["OutputFile"] = output_path
        
    # Replace *.PNG output paths
    if image_path is not None:
        filter_id = get_filter_ids(data, "ITKImageWriter")[0]
        data[filter_id]["FileName"] = image_path
    
    # Write the modified DREAM.3D *.JSON file
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, separators=(",", ": "), sort_keys=True)

# For reasons entirely beyond my comprehension, DREAM.3D does NOT check for attributes from *.DREAM3D files it reads
# Instead, it contains a massive dictionary of expected arrays from the first time the *.DREAM3D file was loaded
# If the previous *.DREAM3D file was changed such that new data exists, DREAM.3D will not read it because it's not in the dictionary
# This function:
#     Reads through the DREAM.3D *.JSON file to find input paths,
#     Scours those *.DREAM3D files for data arrays
#     Adds those data arrays back into the DREAM.3D *.JSON file
################## if there are issues with this function later, it may be because it also needs to scour CellEnsembleData and _SIMPL_GEOMETRY ################
def update_attribute_arrays_expected(path_json):
    
    # import json utils even if called from elsewhere
    new_path = os.path.dirname(__file__)
    if new_path not in sys.path:
        sys.path.append(new_path)
    import utils_json, utils_python
    
    # variables
    path_CellData = "/DataContainers/ImageDataContainer/CellData"
    
    # pull dream3d pipeline
    data_json = utils_json.pull_input(path_json)
    
    # find the data container reader ids
    filter_ids = get_filter_ids(data_json, "DataContainerReader")
    
    for filter_id in filter_ids:

        # Open the path to the *.DREAM3D file in the data container reader
        with h5py.File(data_json[filter_id]["InputFile"], 'r') as file_input:
        
            # Create a list of expected data arrays using the attribute arrays from the previous DREAM.3D files
            data_arrays = []
            for name, properties in zip(file_input[path_CellData].keys(), file_input[path_CellData].values()):
                data_arrays += [{
                    "Component Dimensions": file_input[path_CellData][name].attrs["ComponentDimensions"].tolist()       ,
                    "Flag"                : 2                                                                           , # no idea what flag is
                    "Name"                : name                                                                        ,
                    "Object Type"         : file_input[path_CellData][name].attrs["ObjectType"         ].decode('UTF-8'), # maybe should be "ascii"
                    "Path"                : path_CellData                                                               ,
                    "Tuple Dimensions"    : file_input[path_CellData][name].attrs["TupleDimensions"    ].tolist()       ,
                    "Version"             : file_input[path_CellData][name].attrs["DataArrayVersion"   ].tolist()[0]
                }]
                
        for i, data_container in enumerate(data_json[filter_id]["InputFileDataContainerArrayProxy"]["Data Containers"]):
            for j, attribute_matrix in enumerate(data_json[filter_id]["InputFileDataContainerArrayProxy"]["Data Containers"][i]["Attribute Matricies"]):
                if attribute_matrix["Name"] == "CellData":

                    # replace the expected attribute arrays
                    data_json[filter_id]["InputFileDataContainerArrayProxy"]["Data Containers"][i]["Attribute Matricies"][j]["Data Arrays"] = data_arrays
                    
                    # push inputs to dream3d pipeline
                    utils_json.push_input(path_json, data_json, indent=4)

#dream3d requires non-arbitrary-length strings
def format_string(s):
    ascii_type = h5py.string_dtype('ascii', len(s)+1)
    return np.array(s.encode("utf-8"), dtype=ascii_type)

#map values from local 2d orientations to global 3d orientation
def find_global_values(planes, local_values):
    global_values = []
    for plane in planes:
        global_value = []
        for direction in ["x","y","z"]:
            #this assumes that the orientation in the name matches the orientation in the value
            #if you're having trouble with errors, make sure your names match your actual orientation
            if direction in str(plane).lower():
                global_value.append(local_values[0][0])
                del local_values[0][0]
            else:
                global_value.append(0)
        del local_values[0]
        global_values.append(global_value)
    return np.array(global_values)

#find common value in global 3d orientation given values in local 2d orientations
def find_common_global_value(planes, local_values):
    global_values = find_global_values(planes, local_values)
    global_value = np.zeros((2,3))
    i = 0
    for direction in global_values.transpose():
        direction = direction[np.nonzero(direction)]
        if not all(direction == direction[0]):
            conflicting_planes = np.array(planes)[np.nonzero(direction)]
            direction_names = ["x","y","z"]
            error_message = "Global values do not match: Planes "+str.lower(str(conflicting_planes))+" disagree in "+direction_names[i]+" direction: "+str(direction)
            return error_message
        global_value[:,i] = direction
        i += 1
    return global_value[0,:]

#determine if multiple list-type objects are equal
def lists_are_equal(lists):
    logical_array = np.array([])
    #use chain logic: if x=y and y=z then x=z 
    for i in range(len(lists)-1):
        #convert list-type objects to numpy arrays and flatten to check equality. will return false if out of order
        logical_array = np.append(logical_array, np.array(lists[i]).flatten() == np.array(lists[i+1]).flatten())
    return all(logical_array)

#determine if multiple hdf5 groups are equal
def groups_are_equal(items):
    attributes = []
    isdataset  = []
    for item in items:
        attributes.append(item.attrs)
        isdataset .append(isinstance(item, h5py.Dataset))
    #check attributes
    if not lists_are_equal(attributes):
        print("attributes unequal")
        return False
    #check dataset values
    if all(isdataset):
        data = []
        for item in items:
            data.append(item[()]) #must be [()] for multi-dimensional instead of [:]
        if not lists_are_equal(data):
            print("datasets unequal")
            for i in data:
                print(i)
            return False
        return True
    #check for mismatched types
    elif any(isdataset):
        print("dataset types unequal")
        return False
    #must be a group
    keys = []
    for item in items:
        keys.append(item.keys())
    #check that the keys are equal
    if not lists_are_equal(keys):
        print("keys unequal")
        return False
    #check for empty group
    if len(keys) == 0:
        return True
    #keys must be equal, grab the first set
    keys = keys[0]
    #check subgroups recursively
    isequal = []
    for key in keys:
        sub_items = []
        for item in items:
            sub_items.append(item[key])
        isequal.append(groups_are_equal(sub_items))
    return all(isequal)
