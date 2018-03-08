import numpy as np
import cv2
import os
from tqdm import tqdm
import json_tricks
from os.path import abspath, basename

def split_video(source='1.mp4', directory='tmp', im_name_format="{:05d}.png"):

    cam = cv2.VideoCapture(source)
    # If Camera Device is not opened, exit the program
    if not cam.isOpened():
        print("Video device or file couldn't be opened")
        exit()

    frame_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    print("total frames num:{}".format(frame_length))


    if os.path.isdir(directory):
        in_cmd = input("Seems the directory is not empty, be careful, press `c` to continue, others will terminate the program")
        if not in_cmd == 'c':
            exit()

        print("Using existing dir:{}".format(directory))
    else:
        print("Create dir {} for splited images".format(directory))
        os.makedirs(directory)

    for i in tqdm(range(frame_length)):
        retval, img = cam.read()
        if not retval:
            print ("Cannot capture frame device | CODE TERMINATING :(")
            exit()

        cv2.imwrite(directory+'/'+im_name_format.format(i), img)

    print("Split the video at {} done.".format(directory))


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    image = image.copy()
    assert image.shape[2] == 3 or image.shape[2] == 4
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_blur(image, thres=100):
    """
    Simply detect the image is blur or not, ref https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    The `thres` is somewhat casually picked, may need better analyse the images
    """
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian

    v_l = variance_of_laplacian(image)
    if v_l <= thres:
        return True

    return False


def generate_des(data_dir='', file_name='1.png', des_dir='des'):
    abs_file_name = data_dir+'/'+file_name
    print("Deal with {}".format(abs_file_name))
    assert os.path.isfile(abs_file_name)
    assert os.path.isdir(des_dir)

    im = cv2.imread(abs_file_name)
    v_l = variance_of_laplacian(im)

    des_f = des_dir + '/' + os.path.basename(file_name)[:-4] + '.json'

    info = dict()
    info['data_dir_abs'] = os.path.abspath(data_dir)
    info['file_name'] = file_name
    info['variance_of_laplacian'] = v_l

    with open(des_f, 'w') as f:
        json_tricks.dump(info, f, sort_keys=True, indent=4)

def generate_all_filenames(data_dir):
    files = [f for f in os.listdir(data_dir) if os.path.isfile(data_dir+'/'+f)]
    return files

def eval_blur(des_dir, eval_blur_flag=False, blur_thres=30, visualize=False):
    des_files = generate_all_filenames(des_dir)
    des_files = [f for f in des_files if f[-4:]=='json']
    print(len(des_files))

    blur_l = []
    blur_num = 0
    for f in des_files:
        f = des_dir + '/' + f
        with open(f, 'r') as f_r:
            info = json_tricks.load(f_r)

        blur_l.append(info['variance_of_laplacian'])


        if eval_blur_flag:

            im = cv2.imread(info['data_dir_abs'] + '/' + info['file_name'])
            im_blur = is_blur(im, blur_thres)

            if im_blur:
                blur_num += 1
                if visualize:
                    cv2.imshow('BLUR', im)
                    cv2.waitKey(20)

            with open(f, 'w') as f_w:
                info['is_blur'] = im_blur
                json_tricks.dump(info, f_w, sort_keys=True, indent=4)


    import matplotlib.pyplot as plt
    plt.hist(blur_l, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()

    print("BLUR RATIO:{}".format(blur_num/len(blur_l)))

def genera_files_for_labeling(des_dir):
    des_files = generate_all_filenames(des_dir)
    des_files = [f for f in des_files if f[-4:] == 'json']
    print(len(des_files))

    no_blur_l = []
    blur_l = []

    for f in des_files:
        f = des_dir + '/' + f
        with open(f, 'r') as f_r:
            info = json_tricks.load(f_r)

        if info['is_blur']:
            blur_l.append(f)
        else:
            no_blur_l.append(f)

    no_blur_l = no_blur_l[::20]
    blur_l = blur_l[::10]
    return no_blur_l, blur_l


if __name__ == '__main__':
    v_f = 'C:/Users/miao/Pictures/cam_data/001.mp4'
    # split_video(source=v_f, directory='tmp_001')

    data_dir = 'tmp_001'
    des_dir = 'tmp_001_des'
    if not os.path.isdir(des_dir):
        os.makedirs(des_dir)

    all_file = generate_all_filenames(data_dir)
    print(len(all_file))

    # for f in all_file:
    #     generate_des(data_dir=data_dir, file_name=f, des_dir=des_dir)

    # eval_blur(des_dir=des_dir)
    # eval_blur(des_dir, eval_blur_flag=True, blur_thres=20, visualize=True)

    no_blur_l, blur_l = genera_files_for_labeling(des_dir)
    print(len(no_blur_l), len(blur_l))



