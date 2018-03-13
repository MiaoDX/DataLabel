import os
import subprocess
from subprocess import TimeoutExpired, PIPE



dlib_example_build_dir = '/home/miao/icra2018_dji/dlib/examples/build_release/'
nn_inference_exe = 'dnn_mmod_find_cars2_ex'


model_file = '/home/miao/dataset/cnn_test/mmod_network_00001.dat'

im_file = '/home/miao/dataset/video_002_half/00099.png'


im_dir = '/home/miao/dataset/video_002/frames/'

allfile = [im_dir + '/' + f for f in os.listdir(im_dir)]
allfile = sorted(allfile)

allfile = allfile[::10]

print(len(allfile))
print(allfile[:10])

detected_num = 0
for i, f in enumerate(allfile):

    print(f)
    proc = subprocess.Popen( [os.path.join(dlib_example_build_dir, nn_inference_exe), model_file, f ], stdout=PIPE )
    try:
        outs, errs = proc.communicate(timeout=15)
    except TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()

    outs = outs.decode("utf-8", "ignore")
    # print(outs)
    # print(type(outs))
    if len(outs) > 0:
        detected_num += 1
        print("{}/{}".format(detected_num, i+1))

        outs_rect = outs.split()
        outs_rect = [int(i) for i in outs_rect]
        print(outs_rect)

print("Detected {}/{}".format(detected_num, len(allfile)))