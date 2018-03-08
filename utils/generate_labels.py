import os
import json_tricks
from lib_transfer import faked_labelme_json
from preprocess import genera_files_for_labeling


if __name__ == '__main__':

    data_dir = 'tmp_001'
    des_dir = 'tmp_001_des'

    manual_labeled_dir = 'manual_labeled'
    if not os.path.isdir(manual_labeled_dir):
        os.makedirs(manual_labeled_dir)

    no_blur_l, blur_l = genera_files_for_labeling(des_dir)
    print(len(no_blur_l), len(blur_l))

    print(no_blur_l[:5])

    """
    "data_dir_abs": "H:\\projects\\icra_robomaster\\codes\\DataLabel\\utils\\tmp_001",
    "file_name": "00000.png",
    "is_blur": false,
    "variance_of_laplacian": 187.6600220082495
    """

    for f in no_blur_l[:5]:
        with open(f, 'r') as f_r:
            info = json_tricks.load(f_r)

        im_file = info['data_dir_abs'] + '/' + info['file_name']
        json_file = manual_labeled_dir+'/'+info['file_name'][:-4]+'.json'

        faked_labelme_json(im_file, json_file, label_name="armer")