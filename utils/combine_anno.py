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


def change_file_prefix(old_im_f, old_prefix, target_prefix, win2linux=True):
    if win2linux:
        f_old = pathlib.PureWindowsPath
        f_target = pathlib.PurePosixPath

    else:
        f_old = pathlib.PurePosixPath
        f_target = pathlib.PureWindowsPath

    old_to_target = lambda x: str(f_target(f_old(x)))

    old_im_f = old_to_target(old_im_f)
    old_prefix = old_to_target(old_prefix)
    new_prefix = str(f_target(target_prefix))

    print(old_im_f, old_prefix, new_prefix)
    im_f = old_im_f.replace(old_prefix, new_prefix)
    print(im_f)
    return im_f

def change_des_base_folder(des_folder, old_base_folder, new_base_folder, win2linux=True):
    des_all_files = preprocess.generate_all_abs_filenames(des_folder)
    for des_f in des_all_files:
        print("{} ...".format(des_f))
        with open(des_f, 'r') as f:
            info = json_tricks.load(f)

        abs_file_name = info['abs_file_name'] # type: str
        abs_file_name = change_file_prefix(abs_file_name, old_base_folder, new_base_folder, win2linux=True)
        info['abs_file_name'] = abs_file_name

        with open(des_f, 'w') as f:
            json_tricks.dump(info, f, sort_keys=True, indent=4)


def assert_voc_xml_file_valid(voc_xml_f):

    from lxml.etree import _Element
    from lxml import etree

    tree = etree.parse(voc_xml_f)
    root1 = tree.getroot()  # type: _Element

    # a = root1.find('filename').text
    # print(a)

    s = root1.find('size')
    width = int(s.find('width').text)
    height = int(s.find('height').text)

    obj_l = root1.findall('object')

    for obj in obj_l:
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        try:
            assert xmin < xmax
            assert ymin < ymax
            assert xmax < width
            assert ymax < height
        except AssertionError:
            return False

    return True


def combine_voc_files_pipeline(folders, combine_folder, possible_wrong_folder):
    """
    We need to abandon these with wrong bounding box, https://github.com/thtrieu/darkflow/issues/151
    :return:
    """


    assert not os.path.isdir(combine_folder)
    assert not os.path.isdir(possible_wrong_folder)
    os.makedirs(combine_folder)
    os.makedirs(possible_wrong_folder)

    combine_voc_files(folders, combine_folder)


    all_voc_files = preprocess.generate_all_abs_filenames(combine_folder)

    wrong_files = []
    for voc_f in all_voc_files:
        print(voc_f)
        ok = assert_voc_xml_file_valid(voc_f)
        if not ok:
            wrong_files.append(voc_f)

    from pprint import pprint
    pprint(wrong_files)
    print(len(wrong_files), len(wrong_files) / len(all_voc_files))



    for wrong_f in wrong_files:
        f_basename, f_no_suffix = preprocess.split_the_abs_filename(wrong_f)
        shutil.move(wrong_f, possible_wrong_folder+'/'+f_basename)


    all_voc_files = preprocess.generate_all_abs_filenames(combine_folder)
    wrong_files = []
    for voc_f in all_voc_files:
        print(voc_f)
        ok = assert_voc_xml_file_valid(voc_f)
        if not ok:
            wrong_files.append(voc_f)
    assert len(wrong_files) == 0


def change_folder_prefix_pipeline():

    new_base_folder = '/home/miao/dataset/armer_video/'
    old_base_folder = "H:/projects/icra_robomaster/codes/armer_video/"
    #des_copy = 'H:/projects/icra_robomaster/codes/armer_video_des_win_to_linux/'
    des_copy = '/home/miao/dataset/armer_video_des_win/'

    list_d = ['v001', 'v002', 'v003', 'v004', 'v005']
    des_folders = [des_copy+'/'+d+'/description/' for d in list_d]
    print(des_folders)


    for des_folder in des_folders:
        change_des_base_folder(des_folder, old_base_folder, new_base_folder, win2linux=True)


if __name__ == '__main__':



    # change_folder_prefix_pipeline()




    voc_floder = 'voc_armer_400_half_size'
    d_f = lambda x: "/home/miao/dataset/armer_video/{}/voc_conf/{}/".format(x, voc_floder)
    list_d = ['v001', 'v002', 'v003', 'v004', 'v005']
    folders = [d_f(d) for d in list_d]
    print(folders)

    combine_folder = '/home/miao/dataset/armer_video/voc_all/with_tracking_4000/'
    possible_wrong_folder = '/home/miao/dataset/armer_video/voc_all/with_tracking_4000_possible_wrong/'

    combine_voc_files_pipeline(folders, combine_folder, possible_wrong_folder)