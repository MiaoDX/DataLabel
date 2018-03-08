# The design thoughts

## Pipeline

``` vi
Video -> image frames(png for potential transparent part) -> tags the blurred -> extracted very N frame for labeling -> use existing tracking algorithms to track, both forward and backward -> combine the tracked BBox -> refine BBox(optional) -> train with ML detection algorithms
```

## TODO

* change format to dlib needed
* forward and backward tracking for detection