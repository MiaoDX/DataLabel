from conf.conf_loader import des_dir_conf
from track_for_detection_bbox.track_by_detection import track_with_manual_label_all, show_all_the_bbox
from track_for_detection_bbox.unified_tracking import get_one_tracker

if __name__ == '__main__':

    global_tracker = get_one_tracker('DLIB')

    # track_with_manual_label_all(des_dir_conf, now_split=0, label_name='armer', global_tracker_=global_tracker)

    show_all_the_bbox(des_dir_conf, global_tracker_=global_tracker)