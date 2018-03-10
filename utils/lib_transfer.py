import json
from base64 import b64encode
import numpy as np
import cv2

def faked_labelme_json(im_file, json_file, label_name='AA'):

    with open('labelme_example.json', 'r') as f:
        info = json.load(f)

    with open(im_file, 'rb') as f:
        image_data = f.read()
        image_data = b64encode(image_data).decode('utf-8')

    info['shapes'][0]['label'] = label_name # a little ugly
    info['imagePath'] = im_file
    info['imageData'] = image_data


    with open(json_file, 'w') as f:
        json.dump(info, f, ensure_ascii=True, indent=2)

def generate_dlib_for_one(BBox: np.ndarray, im_file:str):

    if BBox==None or len(BBox) == 0:
        return """
        <image file='{}'>
        </image>
        """.format(im_file)


    assert len(BBox) == 4 and len(BBox.flatten()) == 8
    x0, y0, delta_x, delta_y = cv2.boundingRect(BBox)

    return """
    <image file='{}'>
    <box top='{}' left='{}' width='{}' height='{}'/>
    </image>
    """.format(im_file, x0, y0, delta_x, delta_y)


def write_dlib_detection_file(BBox_list, im_file_list):

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

    content = []
    content.append(start)

    for BBox, im_file in zip(BBox_list, im_file_list):
        content.append(generate_dlib_for_one(BBox, im_file))

    content.append(end)

    return content



if __name__ == '__main__':

    im_file = 'tt/00000.png'
    json_file = 'tt/0_test.json'

    faked_labelme_json(im_file, json_file, label_name="HAHA")