# BBox from tracking

**python 2** only for MTF tracker.

This contains the codes of drawing bounding boxes with tracking method. Code largely copied from [bikz05/object-tracker](https://github.com/bikz05/object-tracker). Instead of using dlib for tracking, for a better tracking performance, we use python binding of [abhineet123/MTF](https://github.com/abhineet123/MTF) for its ability of tracking non-rectangle object, which will save us lots of unnecessary manual labeling.

We can use various backend for tracking, OpenCV, Dlib or MTF.