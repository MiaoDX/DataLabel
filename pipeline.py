import os
import json_tricks
from utils.lib_transfer import faked_labelme_json
from utils.preprocess import split_video, generate_init_des, eval_blur_or_write_blur_flag_to_des, genera_files_for_manual_labeling_for_labelme, split_the_abs_filename, generate_all_abs_filenames
from utils.track_for_detection import update_des_with_manual_or_track_labeled_bbox
import argparse
import random

from conf.conf_loader import *

if __name__ == '__main__':


    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split", help="Perform video splitting", action="store_true")
    parser.add_argument("-i", "--init", help="Generate the initial description files", action="store_true")
    parser.add_argument("-b", "--blureval", help="Eval the blur of frames", action="store_true")
    parser.add_argument("-u", "--updateblur", help="Update the blur flag", action="store_true")

    parser.add_argument("-m", "--manuallabel", help="Generate manual label data to be used for labelme lib",
                        action="store_true")

    parser.add_argument("-x", "--update_des_with_manual_label", help="Update the des files with manual label info",
                        action="store_true")
    parser.add_argument("-y", "--update_des_with_track_label", help="Update the des files with tracking label info",
                        action="store_true")



    args = parser.parse_args()

    if args.split:
        assert os.path.isfile(video_file_conf) and not os.path.exists(frame_dir_conf)
        os.makedirs(frame_dir_conf)
        split_video(source=video_file_conf, split_dir=frame_dir_conf, im_name_format="{:06d}.png")
    elif args.init:
        assert not os.path.exists(des_dir_conf)
        os.makedirs(des_dir_conf)
        generate_init_des(frame_dir_conf, des_dir_conf)
    elif args.blureval:
        assert os.path.isdir(des_dir_conf)
        eval_blur_or_write_blur_flag_to_des(des_dir_conf, write_blur_flag_to_des=False, blur_thres=blurthres_conf)
    elif args.updateblur:
        assert os.path.isdir(des_dir_conf)
        eval_blur_or_write_blur_flag_to_des(des_dir_conf, write_blur_flag_to_des=True, blur_thres=blurthres_conf, visualize=True)

    elif args.manuallabel:
        assert os.path.isdir(des_dir_conf) and not os.path.isdir(manual_labeled_dir_conf)
        os.makedirs(manual_labeled_dir_conf)

        no_blur_l, blur_l = genera_files_for_manual_labeling_for_labelme(des_dir_conf, no_blur_step=no_blur_sample_step_conf, blur_step=blur_sample_step_conf)

        print(len(no_blur_l), len(blur_l))

        manual_l = [*no_blur_l, *blur_l]
        random.shuffle(manual_l) # inplace shuffle

        all_file = generate_all_abs_filenames(des_dir_conf)
        manual_maxnum = int(len(all_file)*manual_maxnum_ratio_conf)
        manual_l = manual_l[:manual_maxnum]


        print("NUM of images need labeling:{}".format(len(manual_l)))
        # print(manual_l)

        for f in manual_l:

            with open(f, 'r') as f_r:
                info = json_tricks.load(f_r)

            im_file = info['abs_file_name']

            f_basename, f_no_suffix = split_the_abs_filename(im_file)

            json_file = manual_labeled_dir_conf + '/' + f_no_suffix + '.json'

            faked_labelme_json(im_file, json_file, label_name=label_name_conf)

    elif args.update_des_with_manual_label:
        update_des_with_manual_or_track_labeled_bbox(manual_labeled_dir_conf, des_dir_conf, manual=True)