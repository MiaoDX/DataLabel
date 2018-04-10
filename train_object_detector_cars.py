import os
import sys
import glob
import dlib
from skimage import io
import numpy as np
import cv2

def train(training_xml_path, model_file="detector.svm"):

    assert os.path.isfile(training_xml_path)
    assert not os.path.isfile(model_file)

    # Now let's do the training.  The train_simple_object_detector() function has a
    # bunch of options, all of which come with reasonable default values.  The next
    # few lines goes over some of these options.
    options = dlib.simple_object_detector_training_options()
    # Since faces are left/right symmetric we can tell the trainer to train a
    # symmetric detector.  This helps it get the most value out of the training
    # data.
    options.add_left_right_image_flips = True
    # The trainer is a kind of support vector machine and therefore has the usual
    # SVM C parameter.  In general, a bigger C encourages it to fit the training
    # data better but might lead to overfitting.  You must find the best C value
    # empirically by checking how well the trained detector works on a test set of
    # images you haven't trained on.  Don't just leave the value set at 5.  Try a
    # few different C values and see what works best for your data.
    options.C = 10
    # Tell the code how many CPU cores your computer has for the fastest training.
    options.num_threads = 6
    options.epsilon = 0.001
    options.be_verbose = True

    options.detection_window_size = 4096 #(32, 32)
    # options.upsample_limit = 8


    # This function does the actual training.  It will save the final detector to
    # detector.svm.  The input is an XML file that lists the images in the training
    # dataset and also contains the positions of the face boxes.  To create your
    # own XML files you can use the imglab tool which can be found in the
    # tools/imglab folder.  It is a simple graphical tool for labeling objects in
    # images with boxes.  To see how to use it read the tools/imglab/README.txt
    # file.  But for this example, we just use the training.xml file included with
    # dlib.

    print("Goingt to train ...")
    dlib.train_simple_object_detector(training_xml_path, model_file, options)

    # Now that we have a face detector we can test it.  The first statement tests
    # it on the training data.  It will print(the precision, recall, and then)
    # average precision.
    print("")  # Print blank line to create gap from previous output
    print("Training accuracy: {}".format(
        dlib.test_simple_object_detector(training_xml_path, model_file)))



def dlib_test(test_folder, model_file="detector.svm"):
    from utils.preprocess import  generate_all_abs_filenames
    # Now let's use the detector as you would in a normal application.  First we
    # will load it from disk.
    detector = dlib.simple_object_detector(model_file)

    # We can look at the HOG filter we learned.  It should look like a face.  Neat!
    win_det = dlib.image_window()
    win_det.set_image(detector)

    # Now let's run the detector over the images in the faces folder and display the
    # results.
    print("Showing detections on the images in the testing folder...")
    win = dlib.image_window()
    files = generate_all_abs_filenames(test_folder)
    for f in files:
        # print("Processing file: {}".format(f))

        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # img = cv2.resize(img, None, fx=0.5, fy=0.5)

        # img = io.imread(f)
        # img = cv2.resize(img, (640, 640))
        # if np.mean(img) > 2.0:
        #     print("Going to dividie 255")
        #     img = img/255.0

        dets = detector(img)
        # print("Number of faces detected: {}".format(len(dets)))
        # for k, d in enumerate(dets):
        #     print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #         k, d.left(), d.top(), d.right(), d.bottom()))

        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(dets)
        # dlib.hit_enter_to_continue()
        # import time
        # time.sleep(0.001)


def show_with_cv_one(img):

    # We can look at the HOG filter we learned.  It should look like a face.  Neat!
    # win_det = dlib.image_window()
    # win_det.set_image(detector)

    import time
    start_time = time.clock()

    img = cv2.resize(img, None, fx=0.5, fy=0.5)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dets = detector(img)

    # print("Number of faces detected: {}".format(len(dets)))

    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        rect = [[d.left(), d.top()], [d.right(), d.top()], [d.right(), d.bottom()], d.left(), d.bottom()]

        print(rect)
        img = cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0,0,255))

    end_time = time.clock()
    # compute the tracking fps
    current_fps = 1.0 / (end_time - start_time)

    cv2.putText(img, "FPS:{:5.2f}".format(current_fps), (5, 15),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

def pipeline_inference(img):
    global detector
    img = show_with_cv_one(img)
    cv2.imshow("Dlib detection result", img)
    cv2.waitKey(1)
    output = img
    return output

def combine_dlib_xml(files, combine_f='dlib_xml_total.xml'):
    # merge dlib file together
    # ./imglab --add /home/miao/dataset/armer_video/v001/dlib_conf/manual_label_armer_half_size.xml /home/miao/dataset/armer_video/v002/dlib_conf/manual_label_armer_half_size.xml
    # --add will output merge.xml, and the name cannot be easily changed

    # shuffle
    # ./imglab merged.xml --shuffle

    import shutil
    import subprocess
    from subprocess import TimeoutExpired, PIPE

    for f in files:
        assert os.path.isfile(f)

    tmp_dir = '/tmp/combine_dlib_xml/'
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)
    file1 = tmp_dir+'/file1.xml'
    file2 = tmp_dir+'/file2.xml'

    dlib_imglab_dir = '/home/miao/icra2018_dji/dlib_198/tools/imglab/build/'

    shutil.copyfile(files[0], file1)
    for f in files[1:]:
        shutil.copyfile(f, file2)


        proc = subprocess.Popen( [dlib_imglab_dir+'/imglab', '--add', file1, file2 ], stdout=PIPE )
        try:
            outs, errs = proc.communicate(timeout=15)
        except TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()

        shutil.copyfile('merged.xml', file1)

    shutil.copyfile('merged.xml', combine_f)

    proc = subprocess.Popen([dlib_imglab_dir + '/imglab', '--shuffle', combine_f], stdout=PIPE)
    try:
        outs, errs = proc.communicate(timeout=15)
    except TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()


if __name__ == '__main__':

    from utils.preprocess import split_the_abs_filename

    dlib_f = 'manual_label_armer_half_size.xml'
    d_f = lambda x: "/home/miao/dataset/armer_video/{}/dlib_conf/{}".format(x, dlib_f)
    list_d = ['v001', 'v002', 'v003', 'v004', 'v005']
    xml_files = [d_f(d) for d in list_d]
    print(xml_files)

    combine_f = 'dlib_merge.xml'

    # combine_dlib_xml(files=xml_files, combine_f=combine_f)
    training_xml_path = combine_f
    save_model_file = combine_f[:-4] + '.svm'

    # train(training_xml_path, save_model_file)

    # dlib_test(frame_dir_conf, model_file=save_model_file)

    test_v_dir = '/home/miao/dataset/armer_video/v001'
    test_v_f = test_v_dir+'/001.mp4'

    f_basename, f_no_suffix = split_the_abs_filename(test_v_f)
    video_output = '{}/{}_dlib_detection.mp4'.format(test_v_dir, f_no_suffix)


    # Now let's use the detector as you would in a normal application.  First we
    # will load it from disk.
    detector = dlib.simple_object_detector(save_model_file)

    from moviepy.editor import VideoFileClip
    clip1 = VideoFileClip(test_v_f)
    clip = clip1.fl_image(pipeline_inference)
    clip.write_videofile(video_output, audio=False)