# Label data for ML/DL methods with ease

[Not finished yet!]

## Some existing tools

* [ViTBAT: Video Tracking and Behavior Annotation Tool](https://vitbat.weebly.com/)
    - I found this a little late, it seems did all the things I needed, however, mine is more adjustable -- with source code to tweak with.

* [Mathworks, Define Ground Truth for Image Collections](https://cn.mathworks.com/help/vision/ug/define-ground-truth-for-image-collections.html)
* [Labelbox](https://github.com/Labelbox/Labelbox)
* [FastAnnotationTool](https://github.com/christopher5106/FastAnnotationTool), seems have lots of methods

* [LabelMeAnnotationTool](https://github.com/CSAILVision/LabelMeAnnotationTool), this is huge


* [label-V](https://github.com/innovationgarage/label-V), something I would like to do, maybe can make use of it
* [wkentaro/labelme](https://github.com/wkentaro/labelme), seems nice, no special operations for different platforms

* [dbolkensteyn/vatic.js](https://github.com/dbolkensteyn/vatic.js), one js tool, really impressive

## Blogs

[A Definitive Guide To Build Training Data For Computer Vision](https://hackernoon.com/a-definitive-guide-to-build-training-data-for-computer-vision-1d1d50b4bf07)

## This one

### Design Philosophy

Semi-automatic labeling, label aided by algorithms.

#### Pipeline

``` vi
Video ->
image frames (png for potential transparent part) ->
distinguish the blurred ->
extracted very N frame for labeling ->
use existing tracking algorithms to track, both forward and backward ->
combine the tracked BBox -> refine BBox (optional) ->
train with ML detection algorithms
```

#### NOTES

* the object is out of the scene for tracking
* data format transfer [coco2pascal.py](https://gist.github.com/chicham/6ed3842d0d2014987186)


## TODO

It seems that, we need a UI, really.

Possible lib:

* [jdoepfert/roipoly.py](https://github.com/jdoepfert/roipoly.py), Select a polygonal region of interest (ROI) with python and matplotlib.

### Dependencies

* dlib
* labelme


* scipy
* preprocess
* tqdm
* scikit-image

* json_tricks
* imutils