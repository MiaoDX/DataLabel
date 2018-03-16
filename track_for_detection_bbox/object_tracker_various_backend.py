# Import the required modules

import cv2
import argparse as ap
# import get_four_points
import time
import numpy as np

try:
    from unified_tracking import get_one_tracker
    from get_four_points import run as get_four_points_run
except ImportError:
    from .unified_tracking import get_one_tracker
    from .get_four_points import run as get_four_points_run


def get_tracked_obj(cam):
    frame_id = 0
    print ("Press key `p` to pause the video to start tracking")
    while True:
        # Retrieve an image and Display it.
        retval, img = cam.read()
        frame_id += 1
        if not retval:
            print ("Cannot capture frame device")
            exit()
        if(cv2.waitKey(10)==ord('p')):
            break
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

        cv2.imshow("Image", img)
    cv2.destroyWindow("Image")

    # Co-ordinates of objects to be tracked
    # will be stored in a list named `points`
    # points = get_points.run(img)

    four_corner = get_four_points_run(img)

    return four_corner, img, frame_id


def run(cam, tracker):
    # Create the VideoCapture object
    # cam = cv2.VideoCapture(source)

    # If Camera Device is not opened, exit the program
    if not cam.isOpened():
        print ("Video device or file couldn't be opened")
        exit()

    four_corner, img, id = get_tracked_obj(cam)
    print ("Frame {}, Points in single MTF:{}".format(id, four_corner))
    
    run_with_bbox(cam, tracker, four_corner, img)


def run_with_bbox(cam, tracker, four_corner, img):
    four_corner = np.array(four_corner)
    print(four_corner)

    assert four_corner.shape == (4, 2)

    frame_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)

    tracker.init(img.copy(), four_corner)
    
    # raw_input('Press to continue ...')

    tracking_fps = []
    tracking_bbox_l = []

    frame_id = 0
    while True:
        start_time = time.clock()
        # Read frame from device or file
        retval, img = cam.read()
        frame_id += 1
        if not retval:
            print ("Cannot capture frame device | CODE TERMINATING :(")
            # exit()
            break

        ok, four_corner_tracked = tracker.update(img)

        assert four_corner_tracked.shape == (4, 2)

        tracking_bbox_l.append(four_corner_tracked)

        end_time = time.clock()

        # compute the tracking fps
        current_fps = 1.0 / (end_time - start_time)
        tracking_fps.append(current_fps)


        # draw the tracker location
        # polylines(img, pts, isClosed, color, thickness=None, lineType=None, shift=None)
        img = cv2.polylines(img, [np.array(four_corner_tracked, np.int32)], isClosed=True, color=(255, 0, 0), thickness=5)

        cv2.putText(img, "ID:{:5d}/{:5d} {:5.2f}".format(frame_id, frame_length, current_fps), (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
        # display the image
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) == 27:
            break

    # Relase the VideoCapture object
    cam.release()
    cv2.destroyWindow("Image")

    mean_fps = np.mean(tracking_fps)

    print ('mean_fps: {}'.format(mean_fps))

    return tracking_bbox_l

if __name__ == "__main__":
    # Parse command line arguments
    parser = ap.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('-d', "--deviceID", help="Device ID")
    group.add_argument('-v', "--videoFile", help="Path to Video File")
    parser.add_argument('-l', "--dispLoc", dest="dispLoc", action="store_true")

    group.add_argument("-f", "--faked", help="Use faked camera", action="store_true")
    # args = vars(parser.parse_args())
    args = parser.parse_args()


    if args.faked:
        from faked_VideoCapture import faked_cam_with_start_end
        frame_dir = 'H:/projects/icra_robomaster/codes/armer_video/v001/frames/'
        cam = faked_cam_with_start_end(frame_dir, start=0, end=200, reverse=False)
    else:
        # Get the source of video
        if args["videoFile"]:
            source = args["videoFile"] # H:/projects/SLAM/MTFDATASET/LinTrack/towel/frame%05d.jpg
        else:
            source = int(args["deviceID"])
        cam = cv2.VideoCapture(source)


    tracker = get_one_tracker('MTF')
    print(type(tracker))

    run(cam, tracker)