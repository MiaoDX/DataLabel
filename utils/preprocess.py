import cv2
import os
from tqdm import tqdm
import json_tricks


def split_video(source='1.mp4', split_dir='tmp', im_name_format="{:05d}.png"):

    cam = cv2.VideoCapture(source)
    # If Camera Device is not opened, exit the program
    if not cam.isOpened():
        print("Video device or file couldn't be opened")
        exit()

    frame_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    print("total frames num:{}".format(frame_length))

    for i in tqdm(range(frame_length)):
        retval, img = cam.read()
        if not retval:
            print ("Cannot capture frame device | CODE TERMINATING :(")
            exit()

        cv2.imwrite(split_dir+'/'+im_name_format.format(i), img)

    print("Split the video at {} done.".format(split_dir))


def _variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    image = image.copy()
    assert image.shape[2] == 3 or image.shape[2] == 4
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def _is_blur(_variance_of_laplacian, thres=100):
    """
    Simply detect the image is blur or not, ref https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    The `thres` is somewhat casually picked, may need better analyse the images
    """
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian

    if _variance_of_laplacian <= thres:
        return True

    return False


def generate_init_des(frame_dir, des_dir):
    # assert os.path.isdir(frame_dir) and not os.path.isdir(des_dir)

    all_frame_files = generate_all_abs_filenames(frame_dir)

    for frame_file in all_frame_files:
        print("Deal with {}".format(frame_file))
        f_basename, f_no_suffix = split_the_abs_filename(frame_file)
        des_file = des_dir+'/'+f_no_suffix+'.json'
        des_file = os.path.abspath(des_file)

        assert os.path.isfile(frame_file) and not os.path.isfile(des_file)

        im = cv2.imread(frame_file)
        v_l = _variance_of_laplacian(im)

        info = dict()
        info['abs_file_name'] = frame_file
        info['variance_of_laplacian'] = v_l

        with open(des_file, 'w') as f:
            json_tricks.dump(info, f, sort_keys=True, indent=4)

# def generate_all_filenames(data_dir):
#     files = [f for f in os.listdir(data_dir) if os.path.isfile(data_dir+'/'+f)]
#     return files

def generate_all_abs_filenames(data_dir):
    files = [os.path.abspath(data_dir+'/'+f) for f in os.listdir(data_dir) if os.path.isfile(data_dir+'/'+f)]
    files = sorted(files)
    return files

def split_the_abs_filename(abs_filename):
    f_basename = os.path.basename(abs_filename)
    f_no_suffix = f_basename.split('.')[0]
    return f_basename, f_no_suffix

def eval_blur_or_write_blur_flag_to_des(des_dir, write_blur_flag_to_des=False, blur_thres=30, visualize=False):
    des_files = generate_all_abs_filenames(des_dir)
    print(len(des_files))

    blur_l = []
    blur_num = 0
    for f in des_files:

        with open(f, 'r') as f_r:
            info = json_tricks.load(f_r)

        v_of_l = info['variance_of_laplacian']
        blur_l.append(v_of_l)

        im_blur = _is_blur(v_of_l, blur_thres)

        if im_blur:

            blur_num += 1

            if visualize:
                im = cv2.imread(info['abs_file_name'])
                cv2.imshow('BLUR', im)
                cv2.waitKey(20)

        if write_blur_flag_to_des:

            with open(f, 'w') as f_w:
                info['is_blur'] = im_blur
                json_tricks.dump(info, f_w, sort_keys=True, indent=4)

    print("BLUR RATIO:{}".format(blur_num / len(des_files)))
    import matplotlib.pyplot as plt
    plt.hist(blur_l, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()



def genera_files_for_manual_labeling_for_labelme(des_dir, no_blur_step=20, blur_step=10):
    des_files = generate_all_abs_filenames(des_dir)
    print(len(des_files))

    no_blur_l = []
    blur_l = []

    for f in des_files:

        with open(f, 'r') as f_r:
            info = json_tricks.load(f_r)

        if info['is_blur']:
            blur_l.append(f)
        else:
            no_blur_l.append(f)

    no_blur_l = no_blur_l[::no_blur_step]
    blur_l = blur_l[::blur_step]
    return no_blur_l, blur_l


if __name__ == '__main__':
    v_f = 'C:/Users/miao/Pictures/cam_data/001.mp4'
    split_video(source=v_f, split_dir='tmp_001')

    data_dir = 'tmp_001'
    des_dir = 'tmp_001_des'
    if not os.path.isdir(des_dir):
        os.makedirs(des_dir)

    all_file = generate_all_abs_filenames(data_dir)
    print(len(all_file))

    # for f in all_file:
    #     generate_des(data_dir=data_dir, file_name=f, des_dir=des_dir)

    # eval_blur(des_dir=des_dir)
    # eval_blur(des_dir, eval_blur_flag=True, blur_thres=20, visualize=True)

    no_blur_l, blur_l = genera_files_for_manual_labeling_from_labelme(des_dir)
    print(len(no_blur_l), len(blur_l))



