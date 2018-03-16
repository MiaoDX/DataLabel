import cv2
import json_tricks
import collections
import numpy as np

try:
    from object_tracker_various_backend import run_with_bbox
    from faked_VideoCapture import faked_VideoCapture
    from helper_f import generate_all_abs_filenames, split_the_abs_filename
    from unified_tracking import get_one_tracker
except ImportError:
    from .object_tracker_various_backend import run_with_bbox
    from .faked_VideoCapture import faked_VideoCapture
    from .helper_f import generate_all_abs_filenames, split_the_abs_filename
    from .unified_tracking import get_one_tracker

INFO_ITEM = collections.namedtuple("INFO", ["frame_split_list", "start_bbox", "end_bbox"])

global global_tracker

def track_with_manual_label(frame_list, start_bbox, now_split):

    start_im = cv2.imread(frame_list[0])

    print ("Since the tracking methods can be not so good, so you should specify whether or not to accept it")

    while True:

        cam = faked_VideoCapture(frame_list[1:])
        tracking_bbox_l = run_with_bbox(cam, global_tracker, np.array(start_bbox), start_im)

        print("NOW split:{}".format(now_split))
        info = 'Press `0` to accept this sequence, `1` for next run, `5` to drop this sequence (for tracking is too bad)'
        in_cmd = str(input(info))
        if in_cmd == '0' or in_cmd == '5':
            break
        elif in_cmd == '1':
            continue
        else:
            print ("Wrong input, continue")
            continue

    if in_cmd == '0':
        return True, tracking_bbox_l
    else:
        return False, []

def track_with_manual_label_all(des_dir, now_split=0, label_name='object', global_tracker_=''):
    global global_tracker
    global_tracker = global_tracker_

    info_list = generate_tracking_info(des_dir)

    info_list = info_list[now_split:]

    for info in info_list:
        print("NOW split:{}".format(now_split))

        frame_split_list, start_bbox, end_bbox = info

        r1, forward_bbox_l = track_with_manual_label(frame_split_list, start_bbox, now_split)
        r2, backward_bbox_l = track_with_manual_label(frame_split_list[::-1], end_bbox, now_split)

        now_split += 1

        if r1 == False or r2 == False: # this will make it this sequence unchanged
            continue


        forward_bbox_l = forward_bbox_l[:-1]  # drop the last one, since it is manual labeled
        backward_bbox_l = backward_bbox_l[:-1]

        backward_bbox_l_to_forward = backward_bbox_l[::-1]

        for frame_f, forward_b, backward_b in zip(frame_split_list[1:-1], forward_bbox_l, backward_bbox_l_to_forward):
            print ("NOW file:{}".format(frame_f))
            f_basename, f_no_suffix = split_the_abs_filename(frame_f)
            des_f = des_dir + '/' + f_no_suffix + '.json'
            with open(des_f, 'r') as f:
                info = json_tricks.load(f)

            assert 'manual_label' not in info or not info['manual_label']

            info['manual_label'] = False
            info['track_label'] = True

            info['label_list'] = [label_name]
            info['forward_bbox'] = forward_b
            info['backward_bbox'] = backward_b
            info['BBox_list'] = [(forward_b+backward_b)/2]

            print (info['BBox_list'][0].shape)

            assert info['BBox_list'][0].shape == (4, 2)

            with open(des_f, 'w') as f:
                json_tricks.dump(info, f, sort_keys=True, indent=4)


def generate_tracking_info(des_dir):

    des_files = generate_all_abs_filenames(des_dir)

    INFO_LIST = []

    start_bbox = []
    end_bbox = []
    frame_list_tmp = []

    for d_f in des_files:
        print ("NOW description file:{}".format(d_f))

        with open(d_f, 'r') as d_r:
            info = json_tricks.load(d_r)

            abs_f = info['abs_file_name']
            frame_list_tmp.append(abs_f)

            if 'manual_label' in info and info['manual_label']:
                if len(start_bbox) == 0: # the very first start bbox
                    start_bbox = info['BBox_list'][0] # BBox_l list is one list, we use only one at present
                    frame_list_tmp = [abs_f]
                else:
                    end_bbox = info['BBox_list'][0]
                    INFO_LIST.append(INFO_ITEM(frame_list_tmp, start_bbox, end_bbox))

                    start_bbox = end_bbox
                    frame_list_tmp = [abs_f]

    return INFO_LIST


def show_all_the_bbox(des_dir, global_tracker_):
    global global_tracker
    global_tracker = global_tracker_
    des_files = generate_all_abs_filenames(des_dir)
    frame_length = len(des_files)
    for i, d_f in enumerate(des_files):

        print ("NOW description file:{}".format(d_f))

        with open(d_f, 'r') as d_r:
            info = json_tricks.load(d_r)

        img = cv2.imread(info['abs_file_name'])


        if 'manual_label' not in info: # skip no label files
            img = cv2.putText(img, "ID:{:5d}/{:5d} {} Label".format(i, frame_length, "NO"), (15, 15),
                              cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
            show_time = 2

        elif 'failed_to_track' in info and info['failed_to_track']:
            input('press to continue')
            img = cv2.putText(img, "ID:{:5d}/{:5d} {}".format(i, frame_length, "failed_to_track"), (15, 15),
                              cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
            show_time = 2

        else:
            bbox = info['BBox_list'][0]

            # polylines(img, pts, isClosed, color, thickness=None, lineType=None, shift=None)
            img = cv2.polylines(img, [np.array(bbox, np.int32)], isClosed=True, color=(255, 0, 0), thickness=5)

            if info['manual_label']:
                # putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) -> img
                img = cv2.putText(img, "ID:{:5d}/{:5d} {} Label".format(i, frame_length, "Manual") , (15, 15),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))

            elif info['track_label']:
                forward_bbox = info['forward_bbox']
                backward_bbox = info['backward_bbox']

                img = cv2.polylines(img, [np.array(forward_bbox, np.int32)], isClosed=True, color=(0, 255, 0), thickness=3)
                img = cv2.polylines(img, [np.array(backward_bbox, np.int32)], isClosed=True, color=(0, 0, 255), thickness=3)

                img = cv2.putText(img, "ID:{:5d}/{:5d} {} Label".format(i, frame_length, "Tracking") , (15, 15),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))

            show_time = 20

        # display the image
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)

        if cv2.waitKey(show_time) == 27:
            break



if __name__ == '__main__':

    base_dir = 'H:/projects/icra_robomaster/codes/armer_video/v001/'
    des_dir = base_dir + '/description/'

    global_tracker_ = get_one_tracker('DLIB')

    track_with_manual_label_all(des_dir, now_split=20, label_name='armer', global_tracker_=global_tracker_)

    # show_all_the_bbox(des_dir, global_tracker_)