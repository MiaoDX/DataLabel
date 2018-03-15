import os
import shutil
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

if __name__ == '__main__':

    d_f = lambda x: "/home/miao/dataset/armer_video/{}/voc_conf/manual_label_armer_half_size".format(x)
    list_d = ['v001', 'v002', 'v003', 'v004', 'v005']
    folders = [d_f(d) for d in list_d]
    print(folders)

    new_folder = '/home/miao/dataset/armer_video/voc_all/manual/'

    combine_voc_files(folders, new_folder)