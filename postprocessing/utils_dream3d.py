import h5py, json, os, sys, subprocess, platform, math
import numpy as np
import shlex

#make a spherical mask in a volume
def make_mask_sphere(dims, center=None, radius=None, origin=[0,0,0]):
    
    if center is None:
        center = [.5*dim for dim in dims]
    if radius is None:
        radius = .25*min(dims)
    
    coordinates = np.mgrid[[range(dim) for dim in dims]]
    r = np.sqrt(sum([(x-x0)**2 for x, x0 in zip(coordinates, center)]))
    r[r > radius] = 0
    r[r > 0     ] = 1
    
    dtype = "DataArray<bool>"
    r = r.reshape(dims+[1])
    r = r.astype(get_datatype(dtype))
    
    return r
    
#make a cubic mask in a volume
def make_mask_cube(dims_outer, center=None, dims_inner=None):
    
    if center is None:
        center     = [int(0.5*dim) for dim in dims_outer]
    if dims_inner is None:
        dims_inner = [int(0.5*dim) for dim in dims_outer]
        
    indecies_inner = tuple(np.ogrid[[range(mid-dim//2, mid+dim//2) for dim, mid in zip(dims_inner, center)]])
    cube_outer = np.zeros(dims_outer)
    cube_outer[indecies_inner] = 1
    
    dtype = "DataArray<bool>"
    cube_outer = cube_outer.reshape(dims_outer+[1])
    cube_outer = cube_outer.astype(get_datatype(dtype))
    
    return cube_outer

#returns numpy class of datatype
def get_datatype(datatype):
    if datatype in ["float32", format_string("DataArray<float>"   ), "DataArray<float>"   ]:
        return np.float32
    if datatype in ["int32"  , format_string("DataArray<int32_t>" ), "DataArray<int32_t>" ]:
        return np.int32
    if datatype in ["int32"  , format_string("DataArray<uint32_t>"), "DataArray<uint32_t>"]:
        return np.uint32
    if datatype in ["uint8"  , format_string("DataArray<bool>"    ), "DataArray<bool>"    ]:
        return np.uint8
    if datatype in ["object" , format_string("StringDataArray"    ), "StringDataArray"    ]:
        return None
        
def get_datatype_dream3d(datatype):
    if datatype in ["float32", np.float32]:
        return "DataArray<float>"
    if datatype in ["int32"  , np.int32]:
        return "DataArray<int32_t>"
    if datatype in ["int32"  , np.uint32]:
        return "DataArray<uint32_t>"
    if datatype in ["uint8"  , np.uint8]:
        return "DataArray<bool>"
    if datatype in ["object"]:
        return "StringDataArray"
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
def import_resolution(ebsd_paths, name_planes, path_Geometry):

    size_voxels = []
    for ebsd_path in ebsd_paths:
            
        # Search through all data containers to find reference information
        with h5py.File(ebsd_path, 'r') as ebsd_file:

            # Each image is 2D, so the spacing in Z dimension is inconsequential
            size_voxels += [ebsd_file[path_Geometry+"/"+"SPACING"][:-1].tolist()]
        
    # If the input contains 3 planes, we are trying to reconstruct a 3D microstructure.
    # The side resolutions should be the same in order to reconstruct properly.
    # This function allows us to determine if the side resolutions are compatible
    # And gives us the common resolutions [x,y,z] if they exist
    if len(name_planes) == 3:
        size_voxels = find_common_global_value(name_planes, size_voxels)
        check_error_resolution(size_voxels)
    # If the input contains 1 plane, we're trying to reconstruct a 2D microstructure.
    # No compatibility check is required, and the resolution is given as [x,y,1]
    else:
        size_voxels = size_voxels+[1]

    return size_voxels
    
#filter a string with blacklist and whitelist
def filter_string(string, blacklist=None, whitelist=None):
    if not blacklist is None:
        for entry in blacklist:
            string = string.replace(entry, "")
    if not whitelist is None:
        string = "".join(list(filter(lambda char: char in set(whitelist), string)))
    return string  
        
#check if there is a common voxel size for each side. if there is not, then there will be no way to reconcile the dream3d data with the slicegan data at the end.
def check_error_resolution(size_voxels):
    if isinstance(size_voxels, str):
        raise ValueError(size_voxels)
    
#check if the crystallography matches exactly. if it does not, then there will be no way to reconcile the dream3d data with the slicegan data at the end.
def check_error_crystallography(ebsd_paths, path_CellEnsembleData, orientations_types):

    # If the reference data should contain multiple images crystallographic data
    if len(ebsd_paths) == 3 and is_crystallographic(orientations_types):

        # Open all the ebsd files
        ebsd_files = [h5py.File(ebsd_path, 'r') for ebsd_path in ebsd_paths]
        
        # Gather all the CellEnsembleData groups into a list
        CellEnsembleData = [file[path_CellEnsembleData] for file in ebsd_files]

        # Check if the groups are exactly equal
        result = groups_are_equal(CellEnsembleData)

        # Close all the ebsd files
        [file.close() for file in ebsd_files]
            
        if result:
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
def rotate_all(path_input ,n, path_CellData, path_Geometry):

    with h5py.File(path_input, 'r+') as file_input:

        # rotate n times
        for i in range(n):
            
            # gather geometry data
            dims        = file_input[path_Geometry+"/"+"DIMENSIONS"][:]
            size_voxels = file_input[path_Geometry+"/"+"SPACING"   ][:]

            # swap x and y geometry data
            dims[0], dims[1] = dims[1], dims[0]
            size_voxels[0], size_voxels[1] = size_voxels[1], size_voxels[0]

            # replace geometry data
            file_input[path_CellData].attrs["TupleDimensions"] = dims.astype(np.uint64)
            file_input[path_Geometry+"/"+"DIMENSIONS"][:] = dims.astype(np.uint64)
            file_input[path_Geometry+"/"+"SPACING"   ][:] = size_voxels

            for item in file_input[path_CellData]:

                # replace geometry data
                dims_string = ','.join(['='.join([name,str(value)]) for name,value in zip(['x','y','z'],dims)])
                file_input[path_CellData+"/"+item].attrs["TupleDimensions"      ] = dims.astype(np.uint64)
                file_input[path_CellData+"/"+item].attrs["Tuple Axis Dimensions"] = dims_string

                # rotate data
                data = file_input[path_CellData+"/"+item][:]
                data = remove_empty_z_container(data)
                data = rotate_90_degrees(data, n=n)
                data = replace_empty_z_container(data)
                data = data.astype(get_datatype(file_input[path_CellData+"/"+item].attrs["ObjectType"]))
                file_input[path_CellData+"/"+item][:] = data

# Create an empty DREAM.3D file with just the bare minimum to be recognized by DREAM.3D
def create_file_dream3d(path_output):

    # make sure the path exists
    path = os.path.split(path_output)[0]
    if not os.path.exists(path):
        os.makedirs(path)

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

# # Insert dream3d formatted attribute array with specified data into specified hdf5 group
# # Should should be a numpy array of shape: [z,y,x,components] or [x,components]
# # A missing component dimension can be inferred, but it's best to be explicit
# def insert_attribute_array(
#     path_output,
#     name,
#     data,
#     dtype="DataArray<float>",
#     path_CellData="/DataContainers/ImageDataContainer/CellData",
#     path_Geometry="/DataContainers/ImageDataContainer/_SIMPL_GEOMETRY"
#     ):

#     ## Find dimension and component information
#     # Assume the data is 2 or 3 dimensional
#     if len(data.shape) > 2:
#         # Assume the data is missing the component dimension
#         if not len(data.shape) > 3:
#             data = data.reshape(np.append(data.shape,1).astype(tuple))
#     # Assume the data is 1 dimensional
#     else:
#         # Assume the data is missing the component dimension
#         if not len(data.shape) > 1:
#             data = data.reshape(np.append(data.shape,1).astype(tuple))
#     dims                  = list(data.shape[:-1][::-1])
#     components            = [data.shape[-1]]
#     tuple_axis_dimensions = ','.join(['='.join([name,str(value)]) for name,value in zip(['x','y','z'],dims)])
        
#     with h5py.File(path_output, 'a') as file_output:
        
#         # create the CellData group and its required attributes if it does not exist
#         group = file_output.require_group(path_CellData)
#         attributes = {
#             "AttributeMatrixType": np.uint32([len(file_output[path_Geometry+"/"+"DIMENSIONS"][...])]),
#             "TupleDimensions"    : np.uint64(file_output[path_Geometry+"/"+"DIMENSIONS"][...])
#         }
#         for key, val in zip(attributes.keys(), attributes.values()):
#             if not key in [i for i in file_output[path_CellData].attrs.keys()]:
#                 group.attrs[key] = val
        
#         # export attribute array
#         dataset = group.create_dataset(name, data=data)
#         dataset.attrs["ComponentDimensions"]   = np.uint64(components)
#         dataset.attrs["DataArrayVersion"]      = np.int32([2])
#         dataset.attrs["ObjectType"]            = format_string(dtype)
#         dataset.attrs["Tuple Axis Dimensions"] = format_string(tuple_axis_dimensions)
#         dataset.attrs["TupleDimensions"]       = np.uint64(dims)

def insert_attribute_array(
    path_output,
    path_group,
    name,
    data,
    dtype,
    attribute_matrix_type="Cell",
    linked_numneighbors_dataset=None,
    tuple_dimensions=None
    ):

    ## Find dimension and component information
    if len(data.shape) == 1:
        dims                        = np.asarray(data.shape).astype(np.uint64)
        components                  = np.asarray(1).reshape((1,)).astype(np.uint64)
    else:
        dims                  = np.asarray(data.shape[:-1][::-1]).astype(np.uint64)
        components            = np.asarray(data.shape[-1]).reshape((1,)).astype(np.uint64)
    if not linked_numneighbors_dataset is None:
        dims = tuple_dimensions
        linked_numneighbors_dataset = format_string(linked_numneighbors_dataset)
    else:
        tuple_axis_dimensions = format_string(','.join(['='.join([name,str(value)]) for name,value in zip(['x','y','z'],dims.tolist())]))

    ## Create attributes for the attribute matrix
    group_attributes = {
        "AttributeMatrixType": np.uint32(get_attribute_matrix_id(attribute_matrix_type)),
        "TupleDimensions"    : dims
    }

    ## Create attributes common to all datasets
    dataset_attributes = {
        "ComponentDimensions"  : components,
        "DataArrayVersion"     : np.asarray(2).reshape((1,)).astype(np.int32),
        "ObjectType"           : format_string(dtype),
        "TupleDimensions"      : dims
    }
    # Fill in attributes for special exceptions
    if not linked_numneighbors_dataset is None:
        dataset_attributes["Linked_NumNeighbors_Dataset"] = linked_numneighbors_dataset
    else:
        dataset_attributes["Tuple Axis Dimensions"      ] = tuple_axis_dimensions
    
    ## Write out to file
    with h5py.File(path_output, 'a') as file_output:

        # create the group and its required attributes if it does not exist
        group = file_output.require_group(path_group)
        for key, val in zip(group_attributes.keys(), group_attributes.values()):
            if not key in [i for i in group.attrs.keys()]:
                group.attrs[key] = val
        
        # export attribute array
        dataset = group.create_dataset(name, data=data)
        for key, val in zip(dataset_attributes.keys(), dataset_attributes.values()):
            if not key in [i for i in dataset.attrs.keys()]:
                dataset.attrs[key] = val

def get_attribute_matrix_id(attribute_matrix_type):
    map_attribute_matrix_type_id = {
        "Vertex" 	    :  0,
        "Edge" 	        :  1,
        "Face" 	        :  2,
        "Cell" 	        :  3,
        "VertexFeature" :  4,
        "EdgeFeature"   :  5,
        "FaceFeature"   :  6,
        "CellFeature"   :  7,
        "VertexEnsemble":  8,
        "EdgeEnsemble"  :  9,
        "FaceEnsemble"  : 10,
        "CellEnsemble"  : 11,
        "MetaData" 	    : 12,
        "Generic" 	    : 13,
        "Unknown" 	    :999
    }
    return map_attribute_matrix_type_id[attribute_matrix_type]
        
def make_xdmf(dream3d_path, padding="   ", n_padding=0):
    
    xdmf_path    = dream3d_path.rsplit(".",1)[0]+".xdmf"
    dream3d_name = os.path.split(dream3d_path)[-1]

    def insert_datacontainer(datacontainer, dream3d_name, padding, n_padding):
    
        def insert_geometry(datacontainer, padding, n_padding):
        
            dimensions = " ".join(str(i+1) for i in datacontainer["_SIMPL_GEOMETRY"+"/"+"DIMENSIONS"][...][::-1].astype(int))
            n_dims     = str(len(datacontainer["_SIMPL_GEOMETRY"+"/"+"DIMENSIONS"][...])) 
            origin     = " ".join(str(i+0) for i in datacontainer["_SIMPL_GEOMETRY"+"/"+"ORIGIN"    ][...][::-1].astype(int))
            spacing    = " ".join(str(i+0) for i in datacontainer["_SIMPL_GEOMETRY"+"/"+"SPACING"   ][...][::-1])
        
            string = ""
            string += (0+n_padding)*padding+f"<Topology TopologyType=\"3DCoRectMesh\" Dimensions=\"{dimensions}\">"+"\n"
            string += (0+n_padding)*padding+"</Topology>"+"\n"
            string += (0+n_padding)*padding+"<Geometry Type=\"ORIGIN_DXDYDZ\">"+"\n"
            string += (1+n_padding)*padding+"<!-- Origin  Z, Y, X -->"+"\n"
            string += (1+n_padding)*padding+f"<DataItem Format=\"XML\" Dimensions=\"{n_dims}\">"+"\n"
            string += (2+n_padding)*padding+f"{origin}"+"\n"
            string += (1+n_padding)*padding+"</DataItem>"+"\n"
            string += (1+n_padding)*padding+"<!-- DxDyDz (Spacing/Resolution) Z, Y, X -->"+"\n"
            string += (1+n_padding)*padding+f"<DataItem Format=\"XML\" Dimensions=\"{n_dims}\">"+"\n"
            string += (2+n_padding)*padding+f"{spacing}"+"\n"
            string += (1+n_padding)*padding+"</DataItem>"+"\n"
            string += (0+n_padding)*padding+"</Geometry>"
            
            return string
            
        def insert_attributearray(attributearray, dream3d_name, padding, n_padding):
        
            datatypes_xdmf    = ["Float", "Int", "UChar", "UChar"]
            datatypes_dream3d = ["DataArray<float>","DataArray<int32_t>","DataArray<bool>","DataArray<uint8_t>"]
            
            path          = attributearray.name
            name          = path.rsplit("/",1)[-1]
            shape         = " ".join(str(i+0) for i in attributearray[...].shape)
            attributetype = "Scalar" if attributearray[...].shape[-1] == 1 else "Vector"
            try:
                datatype  = datatypes_xdmf[datatypes_dream3d.index(attributearray.attrs["ObjectType"]   .decode("utf-8"))]
                precision = 4 if attributearray.attrs["ObjectType"]   .decode("utf-8") == "DataArray<float>" else 1
            except AttributeError: # hacky workaround for matlab string attributes
                datatype  = datatypes_xdmf[datatypes_dream3d.index(attributearray.attrs["ObjectType"][0].decode("utf-8"))]
                precision = 4 if attributearray.attrs["ObjectType"][0].decode("utf-8") == "DataArray<float>" else 1
                
            string = ""
            string += (0+n_padding)*padding+f"<Attribute Name=\"{name}\" AttributeType=\"{attributetype}\" Center=\"Cell\">"+"\n"
            string += (1+n_padding)*padding+f"<DataItem Format=\"HDF\" Dimensions=\"{shape}\" NumberType=\"{datatype}\" Precision=\"{precision}\" >"+"\n"
            string += (2+n_padding)*padding+f"{dream3d_name}:{path}"+"\n"
            string += (1+n_padding)*padding+"</DataItem>"+"\n"
            string += (0+n_padding)*padding+"</Attribute>"
            
            return string
            
        name_datacontainer = datacontainer.name.rsplit("/",1)[-1]
            
        string = ""
        string += (0+n_padding)*padding+f"<!-- *************** START OF {name_datacontainer} *************** -->"+"\n"
        string += (0+n_padding)*padding+f"<Grid Name=\"{name_datacontainer}\" GridType=\"Uniform\">"+"\n"
        
        string += insert_geometry(datacontainer, padding, 1+n_padding)+"\n"
        
        for attributematrix in datacontainer:
            if "TupleDimensions" in datacontainer[attributematrix].attrs:
                attributematrix_dims = " ".join(str(i) for i in datacontainer[attributematrix].attrs["TupleDimensions"].astype(int))
                geometry_dims        = " ".join(str(i) for i in datacontainer["_SIMPL_GEOMETRY"+"/"+"DIMENSIONS"][...].astype(int))
                if attributematrix_dims == geometry_dims:
                    name_attributematrix = datacontainer[attributematrix].name.rsplit('/',1)[-1]
                    string += (1+n_padding)*padding+f"<!-- *************** START OF {name_attributematrix} *************** -->"+"\n"
                    for attributearray in datacontainer[attributematrix]:
                        string += insert_attributearray(datacontainer[attributematrix+"/"+attributearray], dream3d_name, padding, 1+n_padding)+"\n"
                    string += (1+n_padding)*padding+f"<!-- *************** END OF {name_attributematrix} *************** -->"+"\n"
          
        string += (0+n_padding)*padding+"</Grid>"+"\n"
        string += (0+n_padding)*padding+f"<!-- *************** END OF {name_datacontainer} *************** -->"
    
        return string
    
    with h5py.File(dream3d_path, 'r') as dream3d_file, open(xdmf_path, 'w') as xdmf_file:
    
        string = ""
        string += (0+n_padding)*padding+"<?xml version=\"1.0\"?>"+"\n"
        string += (0+n_padding)*padding+"<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\"[]>"+"\n"
        string += (0+n_padding)*padding+"<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.2\">"+"\n"
        string += (1+n_padding)*padding+padding+"<Domain>"+"\n"
        
        for datacontainer in dream3d_file["DataContainers"]:
            string += insert_datacontainer(dream3d_file["DataContainers"+"/"+datacontainer], dream3d_name, padding, 2)+"\n"
        
        string += (1+n_padding)*padding+"</Domain>"+"\n"
        string += (0+n_padding)*padding+"</Xdmf>"
        
        xdmf_file.write(string)

#call dream3d file
def call_dream3d(dream3d_path, json_path):

    def quote(string):
        return "\""+string+"\""
    def path(string):
        if platform.system() == "Windows":
            return quote(os.path.realpath(string))
        else:
            return os.path.realpath(string)
    def subpath():
        if platform.system() == "Windows":
            return ""
        else:
            return "/bin"
    def extension():
        if platform.system() == "Windows":
            return ".exe"
        else:
            return ""
    
    cmd = path(dream3d_path+subpath()+"/PipelineRunner"+extension())+" -p "+path(json_path)
    return subprocess.call(shlex.split(cmd))

#replace the file paths in dream3d file because dream3d cannot do relative pathing
def replace_json_paths(json_path,input_path=None,output_path=None,image_path=None):
    
    # Read the DREAM.3D *.JSON file
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    # Replace input paths
    if input_path is not None:

        for filter_type in ["DataContainerReader", "ReadCtfData"]:

            filter_ids = get_filter_ids(data, filter_type)
            if len(filter_ids) > 0:
                data[filter_ids[0]]["InputFile"] = input_path
                break
            
    # Replace *.DREAM3D output paths
    if output_path is not None:
        filter_ids = get_filter_ids(data, "DataContainerWriter")
        data[filter_ids[0]]["OutputFile"] = output_path
        
    # Replace *.PNG output paths
    if image_path is not None:
        filter_ids = get_filter_ids(data, "ITKImageWriter")[0]
        data[filter_ids[0]]["FileName"] = image_path
    
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
    import utils_json
    
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
                    
def update_expected(path_json):
        
    # import json utils even if called from elsewhere
    new_path = os.path.dirname(__file__)
    if new_path not in sys.path:
        sys.path.append(new_path)
    import utils_json
    
    # pull dream3d pipeline
    data_json = utils_json.pull_input(path_json)
    
    
    for filter_id in get_filter_ids(data_json, "DataContainerReader"):
    
        file_input = data_json[filter_id]["InputFile"]
        with h5py.File(file_input, 'r') as file_input:
            expected = create_expected(file_input)
        
        data_json[filter_id]["InputFileDataContainerArrayProxy"] = expected["InputFileDataContainerArrayProxy"]
        
    # push inputs to dream3d pipeline
    utils_json.push_input(path_json, data_json, indent=4)
    
    
def create_expected(items, depth=0):
    
    if   depth == 0: # file
        expected = {
            "InputFileDataContainerArrayProxy": create_expected(items["DataContainers"], depth+1)
        }
    elif depth == 1: # root
        expected = {
            "Data Containers": [
                create_expected(items[item], depth+1) for item in [*items.keys()]
            ]
        }
    elif depth == 2: # data containers
        excluded = ["_SIMPL_GEOMETRY"]
        expected = {
            "Attribute Matricies": [
                create_expected(items[item], depth+1) for item in [*items.keys()] if not item in excluded
            ],
            "Flag"               : 2,
            "Name"               : items.name.rsplit('/',1)[1],
            "Type"               : 0
        }
    elif depth == 3: # attribute matricies
        expected = {
            "Data Arrays": [
                create_expected(items[item], depth+1) for item in [*items.keys()]
            ],
            "Flag"       : 2,
            "Name"       : items.name.rsplit('/',1)[1],
            "Type"       : int(items.attrs["AttributeMatrixType"])
        }
    elif depth == 4: # data sets
        expected = {
            "Component Dimensions": [int(dim) for dim in items.attrs["ComponentDimensions"]],
            "Flag"                : 2,
            "Name"                : items.name.rsplit('/',1)[1],
            "Object Type"         : items.attrs["ObjectType"].decode(),
            "Path"                : items.name.rsplit('/',1)[0],
            "Tuple Dimensions"    : [int(dim) for dim in items.attrs["TupleDimensions"    ]],
            "Version"             : int(items.attrs["DataArrayVersion"])
        }
            
    return expected

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
            error_message = "Global values do not match: Planes "+str(conflicting_planes).lower()+" disagree in "+direction_names[i]+" direction: "+str(direction)
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
    
def find_mask_intersected_grains(featureids,mask):
    
    inside_mask  = np.unique(featureids[mask == 1])
    outside_mask = np.unique(featureids[mask == 0])

    return np.intersect1d(inside_mask, outside_mask)

def find_voxel_fraction_mask_intersected_grain(featureid,featureids,mask):
    
    n_voxels              = np.count_nonzero(featureids[featureids == featureid])
    n_voxels_outside_mask = np.count_nonzero(featureids[np.logical_and(featureids == featureid, mask == 0)])

    return n_voxels_outside_mask/n_voxels

def remove_masked_grains(featureids,mask,threshold):

    import time

    time_start = time.time()
    n_grains = len(np.unique(featureids))

    for i, featureid in enumerate(np.unique(featureids), 1):
        fraction = find_voxel_fraction_mask_intersected_grain(featureid,featureids,mask)

        # print ETA
        if i%25 == 0:
            time_elapsed = time.time()-time_start
            time_per_grain = time_elapsed/i
            ETA = time_per_grain*(n_grains-i)
            hrs = int(ETA // 3600)
            mins = int((ETA / 3600 % 1) * 60)
            print(f"Grain {i} of {n_grains}. Estimated time remaining: {hrs} hours, {mins} minutes")

        # delete the grain if too much is outside the mask
        if fraction >= threshold:
            featureids[featureids == featureid] = 0

    mask = np.ones(featureids.shape)
    mask[featureids == 0] = 0

    return featureids, mask
    
# this attempts to find celldata attribe matricies by matching the attributematrix "TupleDimension"s
# with the _SIMPL_GEOMETRY dimensions
# returns a list of hdf5 paths to the celldata
# Ex: ["/DataContainers/VolumeDataContainer1/CellData","/DataContainers/VolumeDataContainer2/CellData"]
def find_celldata(path_input, datacontainer=None):

    celldata = []
    
    with h5py.File(path_input , 'r') as file_input:
    
        if datacontainer is None:
        
            for datacontainer in file_input["/DataContainers"]:
                
                celldata += find_celldata(path_input, datacontainer)
                
        else:
        
            datacontainer = file_input["/DataContainers"+"/"+datacontainer]
            
            for attributematrix in datacontainer:
                attributematrix = datacontainer[attributematrix]
                
                if \
                    "AttributeMatrixType" in attributematrix.attrs.keys() \
                    and attributematrix.attrs["AttributeMatrixType"] == get_attribute_matrix_id("Cell"):
                    
                    celldata.append(attributematrix.name)
                        
    return celldata
    
# this attempts to find "featureids" fields by searching in celldata attribute matricies
# for data arrays that contain signed 32 bit integers and whose names contain the substring "featureids"
# by default, it finds celldata using find_celldata, celldata can be specified as well
# celldata should be a list of pathnames.
# Ex: ["/DataContainers/VolumeDataContainer1/CellData","/DataContainers/VolumeDataContainer2/CellData"]
# returns a list of hdf5 paths to the featureids
# Ex: ["/DataContainers/VolumeDataContainer1/CellData/FeatureIds","/DataContainers/VolumeDataContainer2/CellData/FeatureIds"]
def find_featureids(path_input, celldata=None, datacontainer=None):
    
    featureids = []
    
    if celldata is None and datacontainer is None:
        celldata = find_celldata(path_input)
    elif celldata is None:
        celldata = find_celldata(path_input, datacontainer=datacontainer)
    # regardless the value of datacontainer, if celldata is specified, it is used
    
    with h5py.File(path_input , 'r') as file_input:
    
        for attributematrix in celldata:
            attributematrix = file_input[attributematrix]
            
            for dataarray in attributematrix:
                dataarray = attributematrix[dataarray]
                
                is_featureid = \
                    "ObjectType" in dataarray.attrs \
                    and dataarray.attrs["ObjectType"].decode() == "DataArray<int32_t>" \
                    and "featureids" in dataarray.name.rsplit("/",1)[-1].lower()
                if is_featureid:
                    
                    featureids.append(dataarray.name)
                    
    return featureids
    
# this attempts to find cellfeaturedata linked to featureids fields by finding
# 1) the number of unique featureids in the featureids field
# 2) the length of the candidate cellfeaturedata "TupleDimensions"
# if these dimensions match, then the two are at least compatible if not intentionally linked
# by default, if finds featureids using find_featureids, but featureids can be specified as well
# featureids should be a list of hdf5 path names.
# Ex: ["/DataContainers/VolumeDataContainer1/CellData/FeatureIds","/DataContainers/VolumeDataContainer2/CellData/FeatureIds"]
# returns a dictionary mapping of each featureid with all the potential cellfeaturedata matches
# Ex: {"[...]/CellData/FeatureIds":["[...]/CellFeatureData1","[...]/CellFeatureData2"], [...]}
def find_map_featureids_cellfeaturedata(path_input, featureids=None, datacontainer=None):
    
    if featureids is None and datacontainer is None:
        featureids = find_featureids(path_input)
    elif featureids is None:
        featureids = find_featureids(path_input, datacontainer=datacontainer)
    # regardless the value of datacontainer, if featureids is specified, it is used
        
    map_featureids_cellfeaturedata = {featureid:[] for featureid in featureids}
    
    with h5py.File(path_input , 'r') as file_input:
    
        for featureid in map_featureids_cellfeaturedata.keys():
            
            featureids = np.unique(file_input[featureid][...])
            dims_cellfeaturedata = np.array([(featureids.size+1, featureids.size)[0 in featureids]]).astype(int)
            
            datacontainer = file_input[featureid.rsplit("/",2)[0]]
            
            for attributematrix in datacontainer:
                attributematrix = datacontainer[attributematrix]

                # skip any attribute matricies that are not flagged as "CellFeature" matricies
                if                                                                                                 \
                    not (                                                                                          \
                        "AttributeMatrixType" in attributematrix.attrs.keys()                                      \
                        and attributematrix.attrs["AttributeMatrixType"] == get_attribute_matrix_id("CellFeature") \
                    ):
                    
                    continue
                
                is_feature_cellfeaturedata = \
                    "TupleDimensions" in attributematrix.attrs \
                    and all(attributematrix.attrs["TupleDimensions"].astype(int) == dims_cellfeaturedata)
                if is_feature_cellfeaturedata:
                
                    map_featureids_cellfeaturedata[featureid].append(attributematrix.name)
                    
    return map_featureids_cellfeaturedata
    
def flatten(input_):

    if type(input_) in [list, tuple]:
    
        output = type(input_)()
    
        for val in input_:
        
            val = flatten(val)
            
            if type(val) in [list, tuple]:
                output = type(input_)([*output, *val])
            else:
                output = type(input_)([*output,  val])
            
        return output
            
    else:
        
        return input_
        
def unique(input_):

    output = []
    
    for val in flatten(input_):
        if val not in output:
            output.append(val)
    
    return(output)

def crop(path_input, path_output, points):

    if not str(type(points)) in ["<class 'list'>", "<class 'tuple'>", "<class 'numpy.ndarray'>"]:
        return

    bounds  = np.asarray(points).T.tolist()
    # technically this is backwards (should be z,y,x)
    # but this way, the paraview indecies are correct
    x, y, z = [bound for bound in bounds]



    with h5py.File(path_input , 'r') as file_input:
        
        # create a new empty file
        create_file_dream3d(
            path_output
        )

        for name_datacontainer in file_input["/DataContainers"]:
            
            # define paths relative to current datacontainer
            path_datacontainer = "/"+"DataContainers"+"/"+name_datacontainer
            path_geometry      = path_datacontainer+"/"+"_SIMPL_GEOMETRY"

            # find the size of the original domain to prevent indexing errors
            size_domain        = file_input[path_geometry+"/"+"DIMENSIONS"][...]
            dims = [min(size_domain[2],x[1])-x[0], min(size_domain[1],y[1])-y[0], min(size_domain[0],z[1])-z[0]]
            
            # find values that will change upon splitting
            geometry = {
                "dims"       : dims,
                "origin"     : file_input[path_geometry+"/"+"ORIGIN" ],
                "size_voxels": file_input[path_geometry+"/"+"SPACING"]
            }
            celldata = find_celldata(path_input, datacontainer=name_datacontainer)
            map_featureids_cellfeaturedata = find_map_featureids_cellfeaturedata(path_input, datacontainer=name_datacontainer)
            cellfeaturedata = unique([val for val in map_featureids_cellfeaturedata.values()])
            attribute_matricies_mutible = celldata+cellfeaturedata+[path_geometry]
            
            ### export values to new file ###
            
            # export _SIMPL_GEOMETRY group
            insert_geometry(
                path_output,
                geometry,
                path_geometry
            )
            
            # export CellData group(s) 
            for path_celldata in celldata:
            
                for name_dataarray in file_input[path_celldata].keys():


                    if any( np.asarray([z[0],y[0],x[0]]) > size_domain ):
                        continue
                    else:
                        data = file_input[path_celldata+"/"+name_dataarray][z[0]:min(size_domain[0],z[1]),y[0]:min(size_domain[1],y[1]),x[0]:min(size_domain[2],x[1]),...]
                    
                    # re-order featureids
                    if path_celldata+"/"+name_dataarray in [key for key in map_featureids_cellfeaturedata.keys()]:
                        # find the featureids that remain after the crop
                        # sort required so that the values in cellfeaturedata matches
                        # np.unique pre-sorts values, no not needed here explicitly
                        features_old          = np.unique(data)
                        # the featureids must have contiguous unique ids 
                        # such that they can be used to reference cellfeaturedata dataarrays
                        # create the new values, but DON'T start at zero unless it already exists in features_old
                        # this is because 0 has a special significance as an error value throughout dream.3d
                        features_new = np.arange((1,0)[0 in data],features_old.size)
                        # cycle through each old and new value as they're already sorted and of the same size
                        # replace any value of data that matches the old value and replace it with the new value
                        for feature_old, feature_new in zip(features_old, features_new): data[data==feature_old]=feature_new
                    
                    type_data = file_input[path_celldata+"/"+name_dataarray].attrs["ObjectType"].decode('UTF-8')
                    
                    insert_attribute_array(
                        path_output   ,
                        path_celldata ,
                        name_dataarray,
                        data          ,
                        type_data     ,
                        "Cell"
                    )
                    
            # export CellFeatureData group(s)
            for path_featureids, paths_cellfeaturedata in zip(map_featureids_cellfeaturedata.keys(), map_featureids_cellfeaturedata.values()):
                
                if any( np.asarray([z[0],y[0],x[0]]) > size_domain ):
                    continue
                else:
                    data = file_input[path_featureids][z[0]:min(size_domain[0],z[1]),y[0]:min(size_domain[1],y[1]),x[0]:min(size_domain[2],x[1]),...]
                featureids_remaining        = (np.append([0],np.unique(data)),np.unique(data))[0 in data]
                
                for path_cellfeaturedata in paths_cellfeaturedata:
                    
                    for name_dataarray in file_input[path_cellfeaturedata]:
                    
                        data                                 = file_input[path_cellfeaturedata+"/"+name_dataarray][featureids_remaining]
                        type_data                            = file_input[path_cellfeaturedata+"/"+name_dataarray].attrs["ObjectType"].decode('UTF-8')

                        if "Linked NumNeighbors Dataset" in [key for key in file_input[path_cellfeaturedata+"/"+name_dataarray].attrs.keys()]:
                            linked_numneighbors_dataset = file_input[path_cellfeaturedata+"/"+name_dataarray].attrs["Linked NumNeighbors Dataset"].decode()
                            tuple_dimensions = file_input[path_cellfeaturedata+"/"+name_dataarray].attrs["TupleDimensions"]
                        else:
                            linked_numneighbors_dataset = None
                            tuple_dimensions = None
                        
                        insert_attribute_array(
                            path_output                ,
                            path_cellfeaturedata       ,
                            name_dataarray             ,
                            data                       ,
                            type_data                  ,
                            "CellFeature"              ,
                            linked_numneighbors_dataset,
                            tuple_dimensions
                        )
            
            # export all other groups
            for name_attributematrix in file_input[path_datacontainer].keys():

                path_attributematrix = path_datacontainer+"/"+name_attributematrix

                if not path_attributematrix in attribute_matricies_mutible:

                    copy_crystallography(
                        path_output,
                        path_input,
                        path_attributematrix,
                        path_attributematrix
                    )
    
    make_xdmf(path_output)

    #Tested working

    #def unique2D_subarray(a):
    #    dtype1 = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    #    b = np.ascontiguousarray(a.reshape(a.shape[0],-1)).view(dtype1)
    #    return a[np.unique(b, return_index=1)[1]]

    #with h5py.File(path_output , 'r') as file_output:

    #    path_celldata = "/DataContainers/ImageDataContainer/CellData"
    #    path_cellfeaturedata = "/DataContainers/ImageDataContainer/CellFeatureData"
    #    
    #    path_featureids = path_celldata+"/"+"FeatureIds"
    #    path_element_eulerangles = path_celldata+"/"+"AvgEulerAngles"
    #    path_feature_eulerangles = path_cellfeaturedata+"/"+"AvgEulerAngles"

    #    featureids = file_output[path_featureids][...]
    #    element_eulerangles = file_output[path_element_eulerangles][...]
    #    feature_eulerangles = file_output[path_feature_eulerangles][...]
    #    
    #    featureids_unique = np.unique(featureids)
    #    
    #    for i in range(featureids_unique.size):
    #    
    #        element_eulerangle_i = element_eulerangles[element_eulerangles.shape[-1]*[True]*featureids==i].reshape((-1,element_eulerangles.shape[-1]))
    #        element_eulerangle_i_unique = unique2D_subarray(element_eulerangle_i)
    #        
    #        feature_eulerangle_i = feature_eulerangles[i]
    #        
    #        if element_eulerangle_i_unique.shape[0] > 1:
    #            print("error! more than one eulerangle at {i}. maybe feature average not selected?")
    #            continue
    #        if not all(element_eulerangle_i_unique[0] == feature_eulerangle_i):
    #            print(f"failed! not equal at {i}")
    #            print("   "+"element: "+f"{element_eulerangle_i_unique[0]}")
    #            print("   "+"feature: "+f"{feature_eulerangle_i}")