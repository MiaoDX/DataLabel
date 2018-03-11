"""
Track the manual labeled bbox for detection
"""

import json_tricks

try:
    from . import preprocess
except ImportError:
    import preprocess

# from preprocess import generate_all_abs_filenames, split_the_abs_filename

def update_des_with_manual_or_track_labeled_bbox(manual_or_track_labeled_dir, des_dir, manual=True):
    all_labeled_file = preprocess.generate_all_abs_filenames(manual_or_track_labeled_dir)

    for labeled_file in all_labeled_file:
        print("NOW labeled file:{}".format(labeled_file))
        f_basename, f_no_suffix = preprocess.split_the_abs_filename(labeled_file)
        des_file = des_dir + '/' + f_no_suffix + '.json'

        with open(des_file, 'r') as f_r:
            info = json_tricks.load(f_r)

        with open(labeled_file, 'r') as f_r:
            labeled_info = json_tricks.load(f_r)


        info['label_l'] = []
        info['BBox_l'] = []

        if manual:
            info['manual_label'] = True
            info['track_label'] = False
            for shape in labeled_info['shapes']:
                info['label_l'].append(shape["label"])
                info['BBox_l'].append(shape["points"])
        else:
            info['manual_label'] = False
            info['track_label'] = True
            for shape in labeled_info: # ['shapes']
                info['label_l'].append(shape["label"])
                info['BBox_l'].append(shape["points"])

        with open(des_file, 'w') as f_w:
            json_tricks.dump(info, f_w, sort_keys=True, indent=4)




if __name__ == '__main__':
    base_dir = 'H:/projects/icra_robomaster/codes/DataLabel/pipeline_test/'

    des_dir = base_dir + 'des_002'
    manual_labeled_dir = base_dir + 'manual_labeled'

    tracking_labeled_dir = base_dir + 'tracking_labeled'
    update_des_with_manual_or_track_labeled_bbox(manual_labeled_dir, des_dir)