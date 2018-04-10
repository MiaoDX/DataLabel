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


def BBox_list_to_boundingRect_list(BBox_list):
    boundingRect_l = []

    for BBox in BBox_list:
        BBox = np.array(BBox, np.int32)
        BBox = perspective.order_points(BBox)

        x0, y0, delta_x, delta_y = cv2.boundingRect(BBox)
        boundingRect_l.append([(x0, y0), (delta_x+x0, delta_y+y0)])

    return boundingRect_l


def generate_dlib_content_for_one_img(BBox_list: np.ndarray, im_file:str, visualize=True):
    print("{} ...".format(im_file))
    boundingRect_l = BBox_list_to_boundingRect_list(BBox_list)

    if visualize:
        im = cv2.imread(im_file)

        for rect in boundingRect_l:
            cv2.rectangle(im, rect[0], rect[1], (0,0,255))

        cv2.imshow('A', im)
        cv2.waitKey(10)

    content = ["""<image file='{}'>\n""".format(im_file)]
    for rect in boundingRect_l:
        x0, y0 = rect[0]
        x1, y1 = rect[1]
        content.append("""<box top='{}' left='{}' width='{}' height='{}'/>\n""".format(y0, x0, x1-x0, y1-y0)) # note the y0 and x0

    content.append("""</image>\n""")

    return content


def write_content_to_file(content_list, detection_file):
    start = """<?xml version='1.0' encoding='ISO-8859-1'?>
    <?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>
    <dataset>
    <name>Dlib object detection</name>
    <comment>This file is generated for dlib object detection algorithms.</comment>
    <images>
    """
    end = """
    </images>
    </dataset>"""


    with open(detection_file, 'w') as f:
        f.write(start)
        for content in content_list:
            f.writelines(content)
        f.write(end)


def generate_voc_for_one(BBox_list, label_list, im_file, voc_xml_dir):
    assert len(BBox_list) == len(label_list)

    print(im_file)

    im = cv2.imread(im_file)
    height, width, depth = im.shape

    f_basename, f_no_suffix = preprocess.split_the_abs_filename(im_file)

    start = """
<annotation>
<filename>{}</filename>
<size>
    <width>{}</width>
    <height>{}</height>
    <depth>{}</depth>
</size>
<segmented>0</segmented>
    """.format(im_file, width, height, depth) # note the file name now is abs one

    end = """
</annotation>
    """

    boundingRect_l = BBox_list_to_boundingRect_list(BBox_list)

    middle_info = []

    for label_name, rect in zip(label_list, boundingRect_l):
        x0, y0 = rect[0]
        x1, y1 = rect[1]

        middle = """
    <object>
        <name>{}</name>
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>
            """.format(label_name, x0, y0, x1, y1)


        middle_info.append(middle)


    with open(voc_xml_dir+'/'+f_no_suffix+'.xml', 'w') as f:
        f.write(start)
        for middle in middle_info:
            f.writelines(middle)
        f.write(end)




def generate_dlib_and_voc_detection_file_from_des(des_dir,
                                                  sample_num=100000000,
                                                  only_manual_label=False,
                                                  resize=False, resize_ratio=1.0, resized_dir='resized_frame',
                                                  voc_xml_dir='voc_xml_dir',
                                                  dlib_detection_file='dlib_detection.xml'):

    assert os.path.isdir(des_dir)
    assert not os.path.isdir(resized_dir)
    assert not os.path.isdir(voc_xml_dir)
    assert not os.path.isfile(dlib_detection_file)

    if resize:
        os.makedirs(resized_dir)

    all_des_file = preprocess.generate_all_abs_filenames(des_dir)


    im_file_list = []
    BBox_list_list = []
    label_list_list = []

    for des_f in all_des_file:
        print("{} ...".format(des_f))
        with open(des_f, 'r') as f_r:
            info = json_tricks.load(f_r)

        if "manual_label" not in info and "track_label" not in info:
            continue

        abs_file_name = info['abs_file_name']
        bbox_list = np.array(info['BBox_list'])
        label_list = info['label_list']

        label_list_list.append(label_list)
        f_basename, f_no_suffix = preprocess.split_the_abs_filename(abs_file_name)

        if resize:
            im = cv2.imread(abs_file_name)
            im = cv2.resize(im, None, None, resize_ratio, resize_ratio)
            new_file = resized_dir+'/'+f_basename
            cv2.imwrite(new_file, im)

            abs_file_name = os.path.abspath(new_file) #overwrite abs file here
            bbox_list*=resize_ratio

        if only_manual_label:
            if info['manual_label']:
                im_file_list.append(abs_file_name)
                BBox_list_list.append(bbox_list)
        else:
            im_file_list.append(abs_file_name)
            BBox_list_list.append(bbox_list)


    zip_imfile_bbox = list(zip(im_file_list, BBox_list_list))
    import random
    random.shuffle(zip_imfile_bbox)
    zip_imfile_bbox = zip_imfile_bbox[:sample_num]
    zip_imfile_bbox = sorted(zip_imfile_bbox) # sort by im file name

    im_file_list, BBox_list_list = list(zip(*zip_imfile_bbox))

    content_list = []


    for BBox_list, im_file in zip(BBox_list_list, im_file_list):
        content_list.append(generate_dlib_content_for_one_img(BBox_list, im_file, visualize=True))
    cv2.destroyAllWindows()

    write_content_to_file(content_list, dlib_detection_file)



    os.makedirs(voc_xml_dir)
    for BBox_list, label_list, im_file in zip(BBox_list_list, label_list_list, im_file_list):
        generate_voc_for_one(BBox_list, label_list, im_file, voc_xml_dir)




if __name__ == '__main__':
    import sys
    sys.path.append("..")

    from conf.conf_loader import des_dir_conf, dlib_dir_conf, voc_dir_conf, base_folder_conf, des_dir_conf_with_tracking


    if not os.path.isdir(dlib_dir_conf):
        os.makedirs(dlib_dir_conf)

    """
    dlib_detection_file = dlib_dir_conf + '/manual_label_armer.xml'
    voc_xml_dir = voc_dir_conf + '/manual_label'
    generate_dlib_and_voc_detection_file_from_des(des_dir_conf,
                                                  only_manual_label=True,
                                                  voc_xml_dir=voc_xml_dir,
                                                  dlib_detection_file=dlib_detection_file)
    

    dlib_detection_file = dlib_dir_conf + '/manual_label_armer_half_size.xml'
    voc_xml_dir = voc_dir_conf + '/manual_label_armer_half_size'
    resized_dir = base_folder_conf + '/manual_label_armer_half_size'
    if not os.path.isdir(resized_dir):
        os.makedirs(resized_dir)
    generate_dlib_and_voc_detection_file_from_des(des_dir_conf,
                                                  only_manual_label=False,
                                                  voc_xml_dir=voc_xml_dir,
                                                  resize=True, resize_ratio=0.5, resized_dir=resized_dir,
                                                  dlib_detection_file=dlib_detection_file)

    
    dlib_detection_file = dlib_dir_conf + '/label_armer_200.xml'
    voc_xml_dir = voc_dir_conf + '/label_armer_200'
    generate_dlib_and_voc_detection_file_from_des(des_dir_conf,
                                                  sample_num=200,
                                                  only_manual_label=False,
                                                  voc_xml_dir=voc_xml_dir,
                                                  dlib_detection_file=dlib_detection_file)

    dlib_detection_file = dlib_dir_conf + '/label_armer_200_half_size.xml'
    voc_xml_dir = voc_dir_conf + '/label_armer_200_half_size'
    resized_dir = base_folder_conf + '/label_armer_200_half_size'
    if not os.path.isdir(resized_dir):
        os.makedirs(resized_dir)
    generate_dlib_and_voc_detection_file_from_des(des_dir_conf,
                                                  sample_num=200,
                                                  only_manual_label=False,
                                                  resize=True, resize_ratio=0.5, resized_dir=resized_dir,
                                                  voc_xml_dir=voc_xml_dir,
                                                  dlib_detection_file=dlib_detection_file)
    """



    # After tracking
    sampel_num = 400
    dlib_detection_file = dlib_dir_conf + '/dlib_armer_{}_half_size.xml'.format(sampel_num)
    voc_xml_dir = voc_dir_conf + '/voc_armer_{}_half_size'.format(sampel_num)
    resized_dir = base_folder_conf + '/armer_{}_half_size'.format(sampel_num)

    generate_dlib_and_voc_detection_file_from_des(des_dir_conf_with_tracking,
                                                  sample_num=sampel_num,
                                                  only_manual_label=False,
                                                  voc_xml_dir=voc_xml_dir,
                                                  resize=True, resize_ratio=0.5, resized_dir=resized_dir,
                                                  dlib_detection_file=dlib_detection_file)
