import pathlib
import configparser
config = configparser.ConfigParser()



config.read_file(open(str(pathlib.Path(__file__).parent / 'datalabel.cfg')))

base_folder_conf = config['PATH']['base_folder']

video_file_conf = base_folder_conf + config['PATH']['video_file']
frame_dir_conf = base_folder_conf + 'frames/'
des_dir_conf = base_folder_conf + 'description/'
# des_dir_conf_with_tracking = base_folder_conf + 'description_with_tracking/'
manual_labeled_dir_conf = base_folder_conf + 'manual_labeled/'
tracking_labeled_dir_conf = base_folder_conf + 'tracking_labeled/'
dlib_dir_conf = base_folder_conf + 'dlib_conf/'
voc_dir_conf = base_folder_conf + 'voc_conf/'

# for pipeline.py
blurthres_conf = float(config['THRES']['blurthres'])
no_blur_sample_step_conf = int(config['THRES']['no_blur_sample_step'])
blur_sample_step_conf = int(config['THRES']['blur_sample_step'])
manual_maxnum_ratio_conf = float(config['THRES']['manual_maxnum_ratio'])

# general case
label_name_conf = config['LABEL']['label_name']


if __name__ == '__main__':
    print(base_folder_conf)
    print(video_file_conf)