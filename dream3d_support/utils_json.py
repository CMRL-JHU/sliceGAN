import json
import os, sys
new_path = os.path.dirname(__file__)
if new_path not in sys.path:
    sys.path.append(new_path)
import utils_python

def newline_delimited_json_dumps(data, i=1, padding="   "):
    
    # create the structure
    string =                                               \
        "{"+"\n"+               \
        utils_python.pretty_file_dict(data, i=i, padding=padding)+"\n"+ \
        "}"+"\n"
    
    # if this is the master instance, surround with brackets and translate
    if i==1:
        string = map_python_to_json(string)
    
    return string
    
def map_python_to_json(string):
    mapping = [
        ("\'"   , "\""   ),
        ("\\"   , "\\\\" ),
        ("None" , "null" ),
        ("True" , "true" ),
        ("False", "false")
        ]
    return map_replace_string(string, mapping)
     
def map_replace_string(string, mapping):
    for k, v in mapping:
        string = string.replace(k, v)
    return string
    
def pull_input(path_inputs):
    
    # if inputs are scattered through different files
    if type(path_inputs) in [list, tuple]:
        inputs = {}
        for path_input in path_inputs:
            name_input = os.path.split(path_input)[1].replace(".json","")
            inputs[name_input] = pull_input(path_input)
        return inputs

    # if inputs are in the same file
    else:
        with open(path_inputs,"r") as f:
            return json.load(f)

def push_input(path_inputs, inputs, padding=3*" "):
    path = os.path.split(path_inputs)[0]
    if len(path) > 0:
        os.makedirs(path, exist_ok=True)
    with open(path_inputs,"w") as f:
        f.write(newline_delimited_json_dumps(inputs, padding=padding))