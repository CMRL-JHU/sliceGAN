import utils_json
import numpy as np

scale_image = 1/255
shift_image = 0

properties = {}
properties["Quats"] = {
    "n_components"    :4,
    "image_type"      :"colour",
    "datatype_python" :"float32",
    "datatype_dream3d":"DataArray<float>",
    #image data is bounded from [0,255]. divide by 255 to get bounds of [0,1], multiply by 2 to get bounds of [0,2], then subtract 1 to get [-1,1]
    "scale":scale_image * ( 2),
    "shift":shift_image + (-1)
    }
properties["EulerAngles"] = {
    "n_components"    :3,
    "image_type"      :"colour",
    "datatype_python" :"float32",
    "datatype_dream3d":"DataArray<float>",
    #image data is bounded from [0,255]. divide by 255 to get bounds of [0,1], then multiply by 2*pi to get [0,2*pi]
    "scale":scale_image * (2*np.pi),
    "shift":shift_image
    }
properties["Volumes"] = \
properties["EquivalentDiameters"] = \
properties["GBManhattanDistances"] = {
    "n_components"    :1,
    "image_type"      :"grayscale",
    "datatype_python" :"float32",
    "datatype_dream3d":"DataArray<float>",
    "scale":scale_image, ########## this is probably wrong. should scale to some local maximum
    "shift":shift_image
    }
properties["FeatureIds"] = \
properties["Phases"] = {
    "n_components"    :1,
    "image_type"      :"grayscale",
    "datatype_python" :"int32",
    "datatype_dream3d":"DataArray<int32_t>",
    "scale":scale_image, ########## this is probably wrong. should scale to some local maximum
    "shift":shift_image
    }
properties["Mask"] = {
    "n_components"    :1,
    "image_type"      :"grayscale",
    "datatype_python" :"uint8",
    "datatype_dream3d":"DataArray<bool>",
    "scale":scale_image, ########## this is probably wrong. should scale to some local maximum
    "shift":shift_image
    }

utils_json.push_input("properties.json", properties)