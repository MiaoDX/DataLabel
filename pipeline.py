import os
import json_tricks
from utils.lib_transfer import faked_labelme_json
from utils.preprocess import split_video, generate_init_des, eval_blur, genera_files_for_manual_labeling_with_labelme, split_the_abs_filename
from utils.track_for_detection import update_des_with_manual_or_track_labeled_bbox
import argparse
import random

if __name__ == '__main__':


    #base_dir = 'H:/projects/icra_robomaster/codes/DataLabel/video_002/'
    base_dir = '/home/miao/dataset/video_002/'

    video_file = base_dir + '002.mp4'
    frame_dir = base_dir + 'frames'
    des_dir = base_dir + 'description'
    manual_labeled_dir = base_dir + 'manual_labeled'
    tracking_labeled_dir = base_dir + 'tracking_labeled'


    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split", help="Perform video splitting", action="store_true")
    parser.add_argument("-i", "--init", help="Generate the initial description files", action="store_true")
    parser.add_argument("-b", "--blureval", help="Eval the blur of frames", action="store_true")
    parser.add_argument("-u", "--updateblur", help="Update the blur flag", action="store_true")
    parser.add_argument("-f", "--blurthres", help="Threshold for updating the blur flag", type=float, default=30)

    # python pipeline.py --manuallabel --objectname armer
    parser.add_argument("-m", "--manuallabel", help="Generate manual label data to be used for labelme lib",
                        action="store_true")
    parser.add_argument("-x", "--noblurstep", help="No blur sample step for manual label",  type=int, default=30)
    parser.add_argument("-y", "--blurstep", help="Blur sample step for manual label",  type=int, default=20)
    parser.add_argument("-z", "--manualnum", help="Specify the blur sample number for manual label", type=int, default=30)
    parser.add_argument("-n", "--objectname", help="Object name for our dataset", type=str, default="object")

    parser.add_argument("-d", "--updatedeswithmanual", help="Update the des files with manual label info", action="store_true")



    """
    parser.add_argument("-t", "--train", help="Train and save the classifier", action="store_true")
    parser.add_argument("-p", "--predict", metavar="IMAGE_PATH", help="The path to the image file or dir")
    parser.add_argument("-f", "--feature", help="Select the feature to be extracted", default=FEATURES[0], choices=FEATURES)
    parser.add_argument("-c", "--classifier", help="Select the classifier to be used", default=CLASSIFIERS[0], choices=CLASSIFIERS)    
    """

    args = parser.parse_args()

    if args.split:
        assert os.path.isfile(video_file) and not os.path.exists(frame_dir)
        os.makedirs(frame_dir)
        # split_video(source='1.mp4', split_dir='tmp', im_name_format="{:05d}.png")
        split_video(source=video_file, split_dir=frame_dir, im_name_format="{:05d}.png")
    elif args.init:
        assert not os.path.exists(des_dir)
        os.makedirs(des_dir)
        generate_init_des(frame_dir, des_dir)
    elif args.blureval:
        assert os.path.isdir(des_dir)
        eval_blur(des_dir, write_blur_flag_to_des=False)
    elif args.updateblur:
        assert os.path.isdir(des_dir)
        eval_blur(des_dir, write_blur_flag_to_des=True, blur_thres=args.blurthres, visualize=True)

    elif args.manuallabel:
        assert os.path.isdir(des_dir) and not os.path.isdir(manual_labeled_dir)
        os.makedirs(manual_labeled_dir)

        no_blur_l, blur_l = genera_files_for_manual_labeling_with_labelme(des_dir, no_blur_step=args.noblurstep, blur_step=args.blurstep)
        print(len(no_blur_l), len(blur_l))

        manual_l = [*no_blur_l, *blur_l]
        random.shuffle(manual_l) # inplace shuffle
        manual_l = manual_l[:args.manualnum]

        print("NUM of images need labeling:{}".format(len(manual_l)))
        print(manual_l)

        for f in manual_l:

            with open(f, 'r') as f_r:
                info = json_tricks.load(f_r)

            im_file = info['abs_file_name']

            f_basename, f_no_suffix = split_the_abs_filename(im_file)

            json_file = manual_labeled_dir + '/' + f_no_suffix + '.json'

            faked_labelme_json(im_file, json_file, label_name=args.objectname)

    elif args.updatedeswithmanual:
        update_des_with_manual_or_track_labeled_bbox(manual_labeled_dir, des_dir, manual=True)