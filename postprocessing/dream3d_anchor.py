import os, json, warnings, shlex
from pathlib import Path

def is_path_like(string):
    def is_dos_path_like(string):
        return len(list(filter(lambda x: x in string, ["\\\\","\\"]))) > 0 and string.startswith(":",1)
    def is_unix_path_like(string):
        return len(list(filter(lambda x: x in string, ["/"]))) > 0
    return  is_dos_path_like(string) or is_unix_path_like(string)
def replace_separators(string):
    return string.replace("\\\\",os.sep).replace("\\",os.sep).replace('/',os.sep)
def replace_path_parent(file_path,dir_path_parent,dir_name):
    file_path = replace_separators(file_path)
    index = file_path.rfind(os.sep+dir_name+os.sep)
    if index < 0:
        warnings.warn("\nCannot anchor path found outside of directory: "+file_path)
        print("directory path: "+str(dir_path_parent))
        print("directory name: "+str(dir_name))
        return file_path
    file_path = file_path[index:]
    file_path = dir_path_parent+file_path
    return file_path
def find_io_paths(keys):
    whitelist = ["InputFile","InputPath","OutputFile","OutputPath"]
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

def replace_json_paths(json_path,dir_path_parent=None,dir_name=None):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        for filter_ in data:
            io_paths = find_io_paths(data[filter_].keys())
            execute_paths = find_execute_paths(data[filter_].keys())
            if io_paths:
                for io_path in io_paths:
                    data[filter_][io_path] = replace_path_parent(data[filter_][io_path],dir_path_parent,dir_name)
            if execute_paths:
                for execute_path in execute_paths:
                    components = shlex.split(data[filter_][execute_path],posix=False)
                    components = [replace_path_parent(component,dir_path_parent,dir_name) if is_path_like(component) else component for component in components]
                    data[filter_][execute_path] = " ".join(components)
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, separators=(",", ": "), sort_keys=True)

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_parent = str(Path(dir_path).parent.absolute())
dir_name = str(dir_path.replace(dir_path_parent,"")).replace("\\\\","").replace("\\","").replace('/',"")
dream3d_script_names = list(filter(lambda f: f.endswith(".json"), os.listdir(dir_path)))

for dream3d_script_name in  dream3d_script_names:
    dream3d_script_path = (dir_path+os.sep+dream3d_script_name)
    replace_json_paths(dream3d_script_path,dir_path_parent,dir_name)