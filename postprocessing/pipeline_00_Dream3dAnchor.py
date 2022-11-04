import os, json, warnings, shlex
from pathlib import Path

def is_path_like(string):
    def is_dos_path_like(string):
        return len(list(filter(lambda x: x in string, ["\\\\","\\"]))) > 0 and string.startswith(":",1)
    def is_unix_path_like(string):
        return len(list(filter(lambda x: x in string, ["/"]))) > 0
    return  is_dos_path_like(string) or is_unix_path_like(string)
    
def replace_separators(string):
    separators = ["\\\\","\\","/"]
    for separator in separators:
        string = string.replace(separator,os.sep)
    return string
    
def replace_path_parent(file_path,dir_parent,dir_name,dir_name_old=None):
    file_path = replace_separators(file_path)
    if not dir_name_old is None:
        index_new = file_path.rfind(os.sep+dir_name+os.sep)
        index_old = file_path.rfind(os.sep+dir_name_old+os.sep)
        if index_old > index_new:
            index = index_old+len(dir_name_old)+2*len(os.sep)
        elif index_new > index_old:
            index = index_new+len(dir_name    )+2*len(os.sep)
        else:
            index = -1
    else:
        index = file_path.rfind(os.sep+dir_name+os.sep)
        if index >= 0:
            index += len(dir_name)+2*len(os.sep)
    if index < 0:
        warnings.warn(
            "\nCannot anchor path found outside of directory: "+"\n"+
            "   "+file_path
            )
        return file_path
    file_path = file_path[index:]
    file_path = dir_parent+os.sep+dir_name+os.sep+file_path
    return file_path
    
def find_io_paths(keys):
    whitelist = ["InputFile","InputPath","OutputFile","OutputPath","FileName"]
    blacklist = ["DataContainerArrayProxy"]
    return array_filter(keys, whitelist, blacklist)
    
def find_execute_paths(keys):
    whitelist = ["Arguments"]
    return array_filter(keys, whitelist)
    
def array_filter(keys, whitelist = None, blacklist = None):
    if not whitelist is None:
        keys = list(filter(lambda x: list(filter(lambda y: y in x, whitelist)), keys))
    if not blacklist is None:
        keys = list(filter(lambda x: list(filter(lambda y: y not in x, blacklist)), keys))
    return keys

def replace_json_paths(json_path,dir_parent,dir_name,dir_name_old=None):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        for filter_ in data:
            io_paths = find_io_paths(data[filter_].keys())
            execute_paths = find_execute_paths(data[filter_].keys())
            if io_paths:
                for io_path in io_paths:
                    data[filter_][io_path] = replace_path_parent(data[filter_][io_path],dir_parent,dir_name,dir_name_old)
            if execute_paths:
                for execute_path in execute_paths:
                    # not everything in an execute block will be a path, so we have to guess at what may be
                    # ex:
                    #    "/home/user/dream3d/dream3d.exe -p /home/user/do_stuff.json"
                    #    ["/home/user/dream3d/dream3d.exe", "-p", "/home/user/do_stuff.json""]
                    #    [True, False, True]
                    components = shlex.split(data[filter_][execute_path],posix=False)
                    components = [replace_path_parent(component,dir_parent,dir_name,dir_name_old) if is_path_like(component) else component for component in components]
                    data[filter_][execute_path] = " ".join(components)
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, separators=(",", ": "), sort_keys=True)

dir_parent, dir_name = os.path.split(os.path.realpath(__file__+"/"+".."))
dir_name_old = "SliceGAN_Extra.2022-01-19.Cleaned.FeatureAvgQuats"
json_paths = list(filter(lambda f: f.endswith(".json"), os.listdir(".")))

for json_path in  json_paths:
    replace_json_paths(json_path,dir_parent,dir_name,dir_name_old)
