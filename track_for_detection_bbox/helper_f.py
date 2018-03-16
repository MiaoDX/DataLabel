import os

def generate_all_abs_filenames(data_dir):
    files = [os.path.abspath(data_dir+'/'+f) for f in os.listdir(data_dir) if os.path.isfile(data_dir+'/'+f)]
    files = sorted(files)
    return files

def split_the_abs_filename(abs_filename):
    f_basename = os.path.basename(abs_filename)
    f_no_suffix = f_basename.split('.')[0]
    return f_basename, f_no_suffix