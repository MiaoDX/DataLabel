# Import the required modules
import cv2
import argparse
from imutils import perspective
import imutils
import numpy as np




def run(im):
    global pts_list_sorted

    im_disp = im.copy()
    im_draw = im.copy()
    window_name = "Select objects to be tracked here."
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, im_draw)

    # List containing top-left, tr, bl and br to crop the image.
    pts_list = []
    pts_list_sorted = []

    run.mouse_down = False

    def callback(event, x, y, flags, param):
        global pts_list_sorted
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pts_list) > 4:
                print ("There should not be more than 4 points to draw the region to track")
                return

            run.mouse_down = True
            pts_list.append((x, y))
            # pts_list.append([int(x), int(y)])
            print ("Point selected at [{}]".format(pts_list[-1]))

            cv2.circle(im_draw, pts_list[-1], 5, (255, 0, 0))

            print ("pts_list LEN:{}".format(len(pts_list)))

            if len(pts_list) == 4:
                print ("Going to sort")
                pts_list_tmp = np.array(pts_list).reshape(-1, 2)
                pts_list_sorted = perspective.order_points(pts_list_tmp)
                print ("pts_list_sorted:{}".format(pts_list_sorted))

        cv2.imshow(window_name, im_draw)

    cv2.setMouseCallback(window_name, callback)

    print ("Press key `p` to continue with the selected points.")
    print ("Press key `d` to discard the last object selected.")
    print ("Press key `q` to quit the program.")

    colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

    while True:
        # global pts_list_sorted
        # Draw the rectangular boxes on the image
        window_name_2 = "Objects to be tracked."

        if len(pts_list_sorted) == 4:

            box = cv2.minAreaRect(pts_list_sorted) #???????

            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            cv2.drawContours(im_disp, [box], -1, (0, 255, 0), 2)

            # loop over the original points and draw them
            for ((x, y), color) in zip(pts_list_sorted, colors):
                cv2.circle(im_disp, (int(x), int(y)), 5, color, -1)


        # Display the cropped images
        cv2.namedWindow(window_name_2, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name_2, im_disp)

        key = cv2.waitKey(30)

        if key == ord('d'):
            # Press key `d` to delete all the points
            print ("Delte chosen points")
            pts_list_sorted = []

        elif key == ord('p'):
            # Press key `s` to return the selected points
            cv2.destroyAllWindows()
            return pts_list_sorted

        elif key == ord('q'):
            # Press key `q` to quit the program
            print ("Quitting without saving.")
            exit()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imagepath", required=True, help="Path to image")

    args = vars(ap.parse_args())

    try:
        im = cv2.imread(args["imagepath"])
    except:
        print ("Cannot read image and exiting.")
        exit()
        
    points = run(im)
    print ("Rectangular Regions Selected are -> ", points)
