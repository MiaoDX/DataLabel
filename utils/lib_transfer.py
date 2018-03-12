import json
from base64 import b64encode
import numpy as np
import cv2
from pathlib import Path
import json_tricks
from imutils import perspective
import os

try:
    from . import preprocess
except ImportError:
    import preprocess



def faked_labelme_json(im_file, json_file, label_name='AA'):
    example_json = Path(__file__).parent / 'labelme_example.json'
    with open(example_json, 'r') as f:
        info = json.load(f)

    with open(im_file, 'rb') as f:
        image_data = f.read()
        image_data = b64encode(image_data).decode('utf-8')

    info['shapes'][0]['label'] = label_name # a little ugly
    info['imagePath'] = im_file
    info['imageData'] = image_data


    with open(json_file, 'w') as f:
        json.dump(info, f, ensure_ascii=True, indent=2)

def generate_dlib_for_one(BBox: np.ndarray, im_file:str, visualize=True):

    print(im_file)

    if BBox==None or len(BBox) == 0:
        return """
        <image file='{}'>
        </image>
        """.format(im_file)

    BBox = np.array(BBox, np.int32)
    BBox = perspective.order_points(BBox)

    # print(BBox.shape)
    # assert len(BBox) == 4 and len(BBox.flatten()) == 8
    x0, y0, delta_x, delta_y = cv2.boundingRect(BBox)

    if visualize:
        im = cv2.imread(im_file)
        cv2.rectangle(im, (x0, y0), (delta_x+x0, delta_y+y0), (0,0,255))
        cv2.imshow('A', im)
        cv2.waitKey(1)

    return """
    <image file='{}'>
    <box top='{}' left='{}' width='{}' height='{}'/>
    </image>
    """.format(im_file, y0, x0, delta_x, delta_y) # note the y0 and x0


def write_content_to_file(content, detection_file):
    start = """<?xml version='1.0' encoding='ISO-8859-1'?>
    <?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>
    <dataset>
    <name>Training Cars</name>
    <comment>These are images from Udacity class.</comment>
    <images>
    """
    end = """
    </images>
    </dataset>"""



    with open(detection_file, 'w') as f:
        f.write(start)
        f.writelines(content)
        f.write(end)


def generate_dlib_detection_file_from_des(des_dir, only_manual_label=False, change_dir=False, resize=False, sample_num=100,
                                          resize_ratio=1.0, resized_dir='resized_frame', new_frame_dir_preffix = 'I_changed_the_dir', detection_file='detection.xml'):


    all_des_file = preprocess.generate_all_abs_filenames(des_dir)
    BBox_list = []
    im_file_list = []

    for des_f in all_des_file:
        with open(des_f, 'r') as f_r:
            info = json_tricks.load(f_r)

        if "manual_label" not in info and "track_label" not in info:
            continue

        abs_file_name = info['abs_file_name']
        bbox_0 = np.array(info['BBox_l'][0])

        f_basename, f_no_suffix = preprocess.split_the_abs_filename(abs_file_name)

        if resize:
            im = cv2.imread(abs_file_name)
            im = cv2.resize(im, None, None, resize_ratio, resize_ratio)
            new_file = resized_dir+'/'+f_basename
            cv2.imwrite(new_file, im)

            bbox_0*=resize_ratio

        if change_dir:
            abs_file_name = new_frame_dir_preffix+'/'+f_basename


        if only_manual_label:
            if info['manual_label']:
                BBox_list.append(bbox_0)
                im_file_list.append(abs_file_name)
        else:
            BBox_list.append(bbox_0)
            im_file_list.append(abs_file_name)


    content = []

    for BBox, im_file in zip(BBox_list, im_file_list):
        content.append(generate_dlib_for_one(BBox, im_file, visualize=False)) # the trick is not show, will change it latter

    import random
    random.shuffle(content)
    content = content[:sample_num]

    write_content_to_file(content, detection_file)





if __name__ == '__main__':

    # im_file = 'tt/00000.png'
    # json_file = 'tt/0_test.json'
    # faked_labelme_json(im_file, json_file, label_name="HAHA")

    base_dir = 'H:/projects/icra_robomaster/codes/DataLabel/pipeline_test/'

    des_dir = base_dir + 'des_002'
    detection_file = base_dir + 'tmp_armer.xml'

    # generate_dlib_detection_file_from_des(des_dir=des_dir, only_manual_label=False, sample_num=200, detection_file=detection_file)

    # change_dir = 'H:/projects/icra_robomaster/codes/DataLabel/video_resized/frames'

    change_dir = '/home/miao/dataset/video_002_half'
    resized_dir = 'H:/projects/icra_robomaster/codes/DataLabel/video_002_half/frames'


    if not os.path.isdir(change_dir):
        os.makedirs(change_dir)


    generate_dlib_detection_file_from_des(des_dir, only_manual_label=False, change_dir=True, resize=True,
                                          sample_num=200,
                                          resize_ratio=0.5, resized_dir=resized_dir,
                                          new_frame_dir_preffix=change_dir, detection_file='detection_half.xml')

    """
    change_dir = '/home/miao/dataset/video_002'
    generate_dlib_detection_file_from_des(des_dir, only_manual_label=True, change_dir=True, resize=False,
                                          sample_num=200,
                                          resize_ratio=1.0, resized_dir='',
                                          new_frame_dir_preffix=change_dir, detection_file='detection_manual.xml')
    """