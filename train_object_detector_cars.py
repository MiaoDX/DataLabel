import os
import sys
import glob
import dlib
from skimage import io
import numpy as np
import cv2

def train(training_xml_path, model_file="detector.svm"):
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
    options.C = 5
    # Tell the code how many CPU cores your computer has for the fastest training.
    options.num_threads = 6
    options.be_verbose = True

    options.detection_window_size = 4096 #(32, 32)
    # options.upsample_limit = 8

    # training_xml_path = os.path.join(train_folder, "testing_copy.xml") # training
    # testing_xml_path = os.path.join(train_folder, "testing.xml")

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
    # However, to get an idea if it really worked without overfitting we need to
    # run it on images it wasn't trained on.  The next line does this.  Happily, we
    # see that the object detector works perfectly on the testing images.

    # print("Testing accuracy: {}".format(
    #     dlib.test_simple_object_detector(testing_xml_path, "detector.svm")))


def test(test_folder, model_file="detector.svm"):
    # Now let's use the detector as you would in a normal application.  First we
    # will load it from disk.
    detector = dlib.simple_object_detector(model_file)

    # We can look at the HOG filter we learned.  It should look like a face.  Neat!
    win_det = dlib.image_window()
    win_det.set_image(detector)

    # Now let's run the detector over the images in the faces folder and display the
    # results.
    print("Showing detections on the images in the faces folder...")
    win = dlib.image_window()
    for f in glob.glob(os.path.join(test_folder, "*")):
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

def test_show_with_cv(test_folder, model_file="detector.svm"):
    # Now let's use the detector as you would in a normal application.  First we
    # will load it from disk.
    detector = dlib.simple_object_detector(model_file)
    # print(dir(detector))

    # We can look at the HOG filter we learned.  It should look like a face.  Neat!
    win_det = dlib.image_window()
    win_det.set_image(detector)

    # Now let's run the detector over the images in the faces folder and display the
    # results.
    print("Showing detections on the images in the faces folder...")
    # cv2.namedWindow("Tracking")
    for f in glob.glob(os.path.join(test_folder, "*")):
        # print("Processing file: {}".format(f))

        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        dets = detector(img)

        # print("Number of faces detected: {}".format(len(dets)))

        det_rect = []
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            rect = [[d.left(), d.top()], [d.right(), d.top()], [d.right(), d.bottom()], d.left(), d.bottom()]
            det_rect.append(rect)
            print(rect)
            cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0,0,255))

        cv2.imshow('Tracking', img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    return det_rect

if __name__ == '__main__':

    # In this example we are going to train a face detector based on the small
    # faces dataset in the examples/faces directory.  This means you need to supply
    # the path to this faces folder as a command line argument so we will know
    # where it is.

    train_folder = 'H:/projects/icra_robomaster/codes/DataLabel/video_002/'
    preffix = 'detection_manual'

    training_xml_path = train_folder + preffix + '.xml'
    save_model_file = train_folder + preffix + '.svm'

    train(training_xml_path, save_model_file)

    """
    base_dir_test = 'H:/projects/icra_robomaster/codes/DataLabel/video_002/'
    frame_dir = base_dir_test + 'frames'

    # test(frame_dir, model_file=save_model_file)
    test_show_with_cv(frame_dir, model_file=save_model_file)
    """