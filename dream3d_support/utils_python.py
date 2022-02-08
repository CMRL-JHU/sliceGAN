#pip install pickle
import pickle
import os

def dump(obj):
   for attr in dir(obj):
       if hasattr( obj, attr ):
           print( "obj.%s = %s\n" % (attr, getattr(obj, attr)))
           
def pretty_dict(data, title=None, padding="   ", i=1):
    string = ""
    if not title is None:
        string += title+"\n"
    for key, val in zip(data.keys(), data.values()):
        string += i*padding+key+": "
        if type(val) is dict:
            string += "\n"+pretty_dict(val, i=i+1)
        else:
            string += str(val)+"\n"
    if i==1:
        string = string[:-1]
    return string
    
def pretty_file_list(data, i=1, padding="   "):
    # create the structure
    string = ""
    for val in data:
        if type(val) is dict:
            string +=                                               \
                i*padding+"{"+"\n"+               \
                pretty_file_dict(val, i=i+1, padding=padding)+"\n"+ \
                i*padding+"}"+","+"\n"
        elif type(val) is list:
            string +=                                               \
                i*padding+"["+"\n"+                                 \
                pretty_file_list(val, i=i+1, padding=padding)+"\n"+ \
                i*padding+"]"+","+"\n"
        elif type(val) is tuple:
            string +=                                               \
                i*padding+"("+"\n"+                                 \
                pretty_file_tuple(val, i=i+1, padding=padding)+"\n"+ \
                i*padding+")"+","+"\n"
        else:
            string += i*padding
            if type(val) is str:
                string += "\""+val+"\""+","
            else:
                string += str(val)+","
            string += "\n"
                
    # remove unnecessary delimiters (,\n)
    string = string[:-2]
    
    return string
    
def pretty_file_tuple(data, i=1, padding="   "):    
    return pretty_file_list(val, i=i+1, padding=padding)
    
def pretty_file_dict(data, i=1, padding="   "):

    # create the structure
    string = ""
    for key,val in zip(data.keys(),data.values()):
        if type(val) is dict:
            string +=                                                \
                i*padding+"\""+key+"\""+": "+"{"+"\n"+                \
                pretty_file_dict(val, i=i+1, padding=padding)+"\n"+  \
                i*padding+"}"+","+"\n"
        elif type(val) is list:
            string +=                                                \
                i*padding+"\""+key+"\""+": "+"["+"\n"+                \
                pretty_file_list(val, i=i+1, padding=padding)+"\n"+  \
                i*padding+"]"+","+"\n"
        elif type(val) is tuple:
            string +=                                                \
                i*padding+"\""+key+"\""+": "+"("+"\n"+                \
                pretty_file_tuple(val, i=i+1, padding=padding)+"\n"+ \
                i*padding+")"+","+"\n"
        else:
            string += i*padding+"\""+key+"\""+": "
            if type(val) is str:
                string += "\""+val+"\""+","
            else:
                string += str(val)+","
            string += "\n"
                
    # remove unnecessary delimiters (,\n)
    string = string[:-2]
    
    return string
        
def data_load(path_file, name_file):
    if os.path.exists(path_file+"/"+name_file+".p"):
        with open(path_file+"/"+name_file+".p", "rb") as file:
            return pickle.load(file)
    else:
        return None

def data_save(data, path_file, name_file):
    with open(path_file+"/"+name_file+".p", "wb") as file:
        os.makedirs(path_file, exist_ok=True)
        pickle.dump(data, file)
    
def data_copy(root_src, root_dst):
    import shutil, os
    
    if os.path.isdir(root_src):
        for src_dir, dirs, files in os.walk(root_src):
            dst_dir = src_dir.replace(root_src, root_dst, 1)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            for file_ in files:
                src_file = os.path.join(src_dir, file_)
                dst_file = os.path.join(dst_dir, file_)
                if os.path.exists(dst_file):
                    # in case of the src and dst are the same file
                    if os.path.samefile(src_file, dst_file):
                        continue
                    os.remove(dst_file)
                shutil.copy(src_file, dst_dir)
                
    elif os.path.isfile(root_src):
        if not os.path.exists(root_dst):
            os.makedirs(root_dst)
        shutil.copy(root_src, root_dst)
            
    
def dict_to_list(items):
    return [item for item in items]
    
def fill_screen(text="", char=1, pad=" || "):
    if   char == 1:
        char = "_.-'""`-._"
    elif char == 2:
        char = "~*:._.:*"

    # if full with is printed, a new blank line is added
    width_terminal = os.get_terminal_size()[0]-1
    
    if type(text) is str:
        text = [text]
        
    if type(text) in [list, tuple]:
    
        # fill each blurb with padding
        text = [pad+blurb+pad for blurb in text if len(blurb) > 0]
        
        # divide the screen width evenly between filler/text/filler/text/filler, etc...
        fill_length = (width_terminal-len("".join(text)))//(len(text)+1)
        
        # catch width lower than string size
        if fill_length < 0:
            return "".join([blurb for blurb in text])
        
        # create the filler string
        ## add all the full filler strings we can
        fill_string = (fill_length)//len(char)*char
        ## add whatever we can of the filler string to fill in the remaining space
        fill_string += char[0:(fill_length)%len(char)]
        
        # join filler strings and text blurbs together
        string = fill_string+"".join([blurb+fill_string for blurb in text])
        
        # add any missing filler characters from int division
        ## add all the full filler strings we can (probably never used)
        string += (width_terminal-len(string))//len(char)*char
        ## add whatever we can of the filler string to fill in the remaining space
        string += char[0:(width_terminal-len(string))%len(char)]
        
        return string