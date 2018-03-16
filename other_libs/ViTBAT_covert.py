"""
Convert VitBAT annotation info into our description
"""

import json_tricks
import numpy as np
import sys
sys.path.append("..")

from utils.preprocess import generate_all_abs_filenames

def generate_all_VitBAT_bbox(vitbat_file, wanted_id_l=[1,2]):
    """
    k   ID_i   x  y  width  height
    1	1	243.78	213.63	437.43	410.29
    1	2	361.92	4.4911	134.1	57.472
    2	1	242.89	213.18	437.43	410.29
    2	2	348.08	13.963	134.1	59.601

    The frames begin at 1 instead of 0
    And ID_i are different for each object, even though they can be one class
    """
    import numpy as np
    anno_l = np.genfromtxt(vitbat_file, skip_header=True)


    BBox_list_dict = dict()

    for anno in anno_l:

        frame_id, label_id, x0, y0, width, height = anno
        frame_id = int(frame_id) - 1 # the difference between matlab and python

        topleft = [x0, y0]
        topright = [x0+width, y0]
        bottomright = [x0+width, y0+height]
        bottomleft = [x0, y0+height]


        label_id = int(label_id)
        if label_id in wanted_id_l:
            BBox = [topleft, topright, bottomright, bottomleft]

            if frame_id not in BBox_list_dict: # have not init
                BBox_list_dict[frame_id] = [BBox] # note the []

            else:
                BBox_list_dict[frame_id] .append(BBox)

    print(len(BBox_list_dict))
    # print(len(BBox_list_dict[0]))
    print(len(BBox_list_dict[3]))
    print(BBox_list_dict[3])

    return BBox_list_dict


def write_vitbat_to_des(des_folder, BBox_list_dict, label_name):
    all_des_f = generate_all_abs_filenames(des_folder)

    for i, des_f in enumerate(all_des_f):

        print("{} ...".format(des_f))

        with open(des_f, 'r') as f:
            info = json_tricks.load(f)

        if i not in BBox_list_dict: # not in vitbat
            continue

        info['BBox_list'] = np.array(BBox_list_dict[i])
        info['label_list'] = [label_name for _ in range(len(BBox_list_dict[i]))]

        with open(des_f, 'w') as f:
            json_tricks.dump(info, f, sort_keys=True, indent=4)


if __name__ == '__main__':

    f = '002_IndividualStates.txt'

    BBox_list_dict = generate_all_VitBAT_bbox(f)

    des_folder = 'H:/projects/icra_robomaster/codes/armer_video/v001/description2/'

    label_name = 'armer'

    write_vitbat_to_des(des_folder, BBox_list_dict, label_name)

