import os
import shutil
import json_tricks
import pathlib

try:
    import preprocess as preprocess
except ImportError:
    import utils.preprocess as preprocess

def combine_voc_files(floders, new_folder):

    if not os.path.isdir(new_folder):
        os.makedirs(new_folder)

    for i, folder in enumerate(floders):
        voc_xml_files = preprocess.generate_all_abs_filenames(folder)
        for voc_xml_f in voc_xml_files:
            print(voc_xml_f)
            f_basename, f_no_suffix = preprocess.split_the_abs_filename(voc_xml_f)
            new_file = "{}/{:02}__{}".format(new_folder, i, f_basename)
            shutil.copyfile(voc_xml_f, new_file)


def change_des_base_folder(des_folder, old_base_folder, new_base_folder):
    des_all_files = preprocess.generate_all_abs_filenames(des_folder)
    for des_f in des_all_files:
        print("{} ...".format(des_f))
        with open(des_f, 'r') as f:
            info = json_tricks.load(f)

        abs_file_name = info['abs_file_name'] # type: str

        # all change to pathlib first, to make sure the replace is okay to do
        abs_file_name = str(pathlib.Path(abs_file_name))
        old_base_folder = str(pathlib.Path(old_base_folder))
        new_base_folder = str(pathlib.Path(new_base_folder))


        info['abs_file_name'] = abs_file_name.replace(old_base_folder, new_base_folder)

        with open(des_f, 'w') as f:
            json_tricks.dump(info, f, sort_keys=True, indent=4)



if __name__ == '__main__':

    d_f = lambda x: "/home/miao/dataset/armer_video/{}/voc_conf/manual_label_armer_half_size".format(x)
    list_d = ['v001', 'v002', 'v003', 'v004', 'v005']
    folders = [d_f(d) for d in list_d]
    print(folders)

    new_folder = '/home/miao/dataset/armer_video/voc_all/manual/'

    # combine_voc_files(folders, new_folder)

    new_base_folder = '/home/miao/dataset/armer_video/'
    old_base_folder = "H:/projects/icra_robomaster/codes/armer_video/"
    des_copy = 'H:/projects/icra_robomaster/codes/armer_video_des_win_to_linux/'

    list_d = ['v001', 'v002', 'v003', 'v004', 'v005']
    des_folders = [des_copy+'/'+d+'/description/' for d in list_d]
    print(des_folders)


    for des_folder in des_folders:
        change_des_base_folder(des_folder, old_base_folder, new_base_folder)

