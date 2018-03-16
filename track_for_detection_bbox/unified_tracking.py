"""
A common interface for tracking with dlib or MTF
"""
import os
import sys
import cv2
import dlib
import numpy as np
from imutils import perspective

have_pyMTF = True
try:
    import pyMTF
except:
    print("It seems you do not have MTF, won't use it")
    have_pyMTF = False

ocv_tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']

class UniTracker(object):
    """
    Just the opencv ones https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
    """

    def __init__(self, tracker_name, configure_info=''):
        self.tracker = None

    def init(self, frame, four_corner):
        pass

    def update(self, frame):
        pass

class OCVTracker(UniTracker):
    """
    OpenCV Tracker
    """
    
    def __init__(self, tracker_name, configure_info=''):
        super(OCVTracker, self).__init__(tracker_name, configure_info)
        tracker_name = tracker_name.upper()
        assert tracker_name in ocv_tracker_types

        if tracker_name == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        elif tracker_name == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif tracker_name == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif tracker_name == 'TLD':
            tracker = cv2.TrackerTLD_create()
        elif tracker_name == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        elif tracker_name == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        else:
            print("Wrong tracker name for opencv")
            exit()

        self.tracker = tracker


    def init(self, frame, four_corner):

        # Initialize tracker with first frame and bounding box
        x0_y0_width_height = four_ordinary_corner_to_x0_y0_width_height(four_corner)
        ok = self.tracker.init(frame, x0_y0_width_height)
        if not ok:
            print("Failed to init")
            exit()

    def update(self, frame):
        assert self.tracker is not None
        ok, x0_y0_width_height = self.tracker.update(frame)

        if ok:
            return ok, x0_y0_width_height_to_four_corner(x0_y0_width_height)
        else:
            return ok, None


class DlibTracker(UniTracker):

    def __init__(self, tracker_name, configure_info=''):
        super(DlibTracker, self).__init__(tracker_name, configure_info)
        tracker_name = tracker_name.upper()
        assert tracker_name == 'DLIB'

        self.tracker = dlib.correlation_tracker()

    def init(self, frame, four_corner):
        x0_y0_width_height = four_ordinary_corner_to_x0_y0_width_height(four_corner)
        x0, y0, width, height = x0_y0_width_height

        self.tracker.start_track(frame, dlib.rectangle(x0, y0, x0+width, y0+height))

    def update(self, frame):
        assert self.tracker is not None
        # Update the tracker  
        self.tracker.update(frame)
        # Get the position of the object, draw a 
        # bounding box around it and display it.
        rect = self.tracker.get_position()        
        
        pt1 = [int(rect.left()), int(rect.top())]
        pt2 = [int(rect.right()), int(rect.top())]
        pt3 = [int(rect.right()), int(rect.bottom())]
        pt4 = [int(rect.left()), int(rect.bottom())]
        
        four_corner = [pt1, pt2, pt3, pt4]
        four_corner = np.array(four_corner)
        return True, four_corner



    """
    print ("Going to create")
    config_root_dir = 'demo_config/pffclm_500_ssim_lintrack_2'
    use_rgb_input = True

    init_img = img.copy()
    # initialize tracker with the first frame and the initial corners
    if not use_rgb_input:
        init_img = cv2.cvtColor(init_img, cv2.COLOR_RGB2GRAY)
    points = points.T

    if not pyMTF.initialize(init_img.astype(np.uint8), points.astype(np.float64)):
        print ('Tracker initialization was unsuccessful')
        sys.exit()
    """

class MTFTracker(UniTracker):

    def __init__(self, tracker_name, configure_info='demo_config/pffclm_500_ssim_lintrack_2'):
        super(MTFTracker, self).__init__(tracker_name, configure_info)
        tracker_name = tracker_name.upper()
        assert tracker_name == 'MTF'
        assert os.path.isdir(configure_info)

        if not pyMTF.create(configure_info):
            print ('Tracker creation was unsuccessful')
            sys.exit()

        # self.tracker = None
        self.tracker_corners = np.zeros((2, 4), dtype=np.float64) # note, this is (2,4)

    def init(self, frame, four_corner):
        
        if not pyMTF.initialize(frame.astype(np.uint8), four_corner.T.astype(np.float64)): # note the transparent
            print ('Tracker initialization was unsuccessful')
            sys.exit()


    def update(self, frame):

        # update the tracker with the current frame
        pyMTF.update(frame.astype(np.uint8), self.tracker_corners)
        return True, self.tracker_corners.T.copy()



def four_ordinary_corner_to_x0_y0_width_height(four_corner):
    assert four_corner.shape == (4, 2)
    BBox = perspective.order_points(four_corner)

    x0_y0_width_height = cv2.boundingRect(BBox)
    return x0_y0_width_height

def x0_y0_width_height_to_four_corner(x0_y0_width_height):
    x0, y0, width, height = x0_y0_width_height
    topleft = [x0, y0]
    topright = [x0 + width, y0]
    bottomright = [x0 + width, y0 + height]
    bottomleft = [x0, y0 + height]

    BBox = [topleft, topright, bottomright, bottomleft]
    BBox = np.array(BBox)
    assert BBox.shape == (4,2)

    return BBox


def get_one_tracker(tracker_name = 'MTF'):    
    tracker_name = tracker_name.upper()
    if tracker_name in ocv_tracker_types:
        return OCVTracker(tracker_name)
    elif tracker_name == 'DLIB':
        return DlibTracker(tracker_name)
    elif tracker_name == 'MTF':
        global have_pyMTF
        assert have_pyMTF
        return MTFTracker(tracker_name)