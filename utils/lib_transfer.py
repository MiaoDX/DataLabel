import json
from base64 import b64encode

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



if __name__ == '__main__':

    im_file = 'tt/00000.png'
    json_file = 'tt/0_test.json'

    faked_labelme_json(im_file, json_file, label_name="HAHA")