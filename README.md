# To get the images from video run extract_unique_frames_from_video.py 
## Update the video file name and run the code
# Custom Vision Export Object Detection Models

## Prerequisites
(For TensorFlow Lite model) TensorFlow Lite 2.1 or newer

## Input specification
This model expects 320x320, 3-channel RGB images. Pixel values need to be in the range of [0-255].

## Output specification
There are three outputs from this model.

* detected_boxes
The detected bounding boxes. Each bounding box is represented as [x1, y1, x2, y2] where (x1, y1) and (x2, y2) are the coordinates of box corners.
* detected_scores
Probability for each detected boxes.
* detected_classes
The class index for the detected boxes.

