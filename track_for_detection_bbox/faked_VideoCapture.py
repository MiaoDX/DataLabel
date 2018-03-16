# -*- coding:utf-8 -*-

"""
A faked VideoCapture to use image list as input source
"""

import cv2
import os

class faked_VideoCapture(object):

    def __init__(self, file_l=[]):
        self.file_l = file_l
        self.now_num = 0

    def read(self):
        if self.now_num<len(self.file_l):
            im = cv2.imread(self.file_l[self.now_num])
            self.now_num+=1
            return True, im
        else:
            return False, None

    def isOpened(self):
        return len(self.file_l) > 0

    def get(self, flag):
        if flag == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.file_l)

    def release(self):
        return


def faked_cam_with_start_end(frame_dir, start=0, end=10, reverse=False):

    all_files = [os.path.abspath(frame_dir+'/'+f) for f in os.listdir(frame_dir) if os.path.isfile(frame_dir+'/'+f)]
    all_files = sorted(all_files)

    files = all_files[start:end]

    if reverse:
        files = files[::-1]

    print (len(files))

    cam = faked_VideoCapture(files)
    return cam

if __name__ == '__main__':


    frame_dir = 'H:/projects/icra_robomaster/codes/armer_video/v001/frames/'

    cam = faked_cam_with_start_end(frame_dir, end=200)

    # If Camera Device is not opened, exit the program
    if not cam.isOpened():
        print ("Video device or file couldn't be opened")
        exit()

    frame_id = 0
    while True:
        retval, img = cam.read()
        frame_id += 1
        if not retval:
            print ("Cannot capture frame device | CODE TERMINATING :(")
            break

        cv2.imshow('A', img)
        cv2.waitKey(20)

    print ("ALL done")