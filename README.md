 #open cvlib ------> example  ------ > use the updated gender detection.py to verify the input gender(stored in excel) and predicted gender 
 #also creates new excel for the mismatched for easy error detection.
 
 follow bellow steps to install and work with the other funtions.
 

 Computer Vision code for gender and Age detection in Python.

## How to Installation

### Installing dependencies

packages needed , other wise  cvlib is completely pip installable.

* OpenCV and * TensorFlow

use the following command 

`pip install opencv-python tensorflow` 
 
#### For Gpu optimization use :-
Install `tensorflow-gpu` package through `pip`. Nvidia drivers required  (CUDA ToolKit, CuDNN etc). 

 by default cpu-only `tensorflow` package.

### Installing main components 

`pip install cvlib`


## Face detection
Detecting faces in an image done by calling the function `detect_face()`. 
return values will be  bounding box corners and  probability of the prediction  for all the faces detected.
### Example :

```python
import cvlib as cv
faces, confidences = cv.detect_face(image)
```
Seriously, that's all it takes to do face detection with `cvlib`. Underneath it is using OpenCV's `dnn` module with a pre-trained caffemodel to detect faces.

To enable GPU
```python
faces, confidences = cv.detect_face(image, enable_gpu=True)
```

Checkout `face_detection.py` in `examples` directory for the complete code.

### Sample output :

![](examples/images/face_detection_output.jpg)

## Gender detection
Once face is detected, it can be passed on to `detect_gender()` function to recognize gender. It will return the labels (man, woman) and associated probabilities.

### Example

```python
label, confidence = cv.detect_gender(face)
```

Underneath `cvlib` is using an AlexNet-like model trained on [Adience dataset](https://talhassner.github.io/home/projects/Adience/Adience-data.html#agegender) by Gil Levi and Tal Hassner for their [CVPR 2015 ](https://talhassner.github.io/home/publication/2015_CVPR) paper.

To enable GPU
```python
label, confidence = cv.detect_gender(face, enable_gpu=True)
```

Checkout `gender_detection.py` in `examples` directory for the complete code.

### Sample output :

![](examples/images/gender_detection_output.jpg)

## Object detection
Detecting common objects in the scene is enabled through a single function call `detect_common_objects()`. It will return the bounding box co-ordinates, corrensponding labels and confidence scores for the detected objects in the image.

### Example :

```python
import cvlib as cv
from cvlib.object_detection import draw_bbox

bbox, label, conf = cv.detect_common_objects(img)

output_image = draw_bbox(img, bbox, label, conf)
```
Underneath it uses [YOLOv4](https://github.com/AlexeyAB/darknet) model trained on [COCO dataset](http://cocodataset.org/) capable of detecting 80 [common objects](https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.txt) in context.

To enable GPU
```python
bbox, label, conf = cv.detect_common_objects(img, enable_gpu=True)
```

Checkout `object_detection.py` in `examples` directory for the complete code.

### Real time object detection
`YOLOv4` is actually a heavy model to run on CPU. If you are working with real time webcam / video feed and doesn't have GPU, try using `tiny yolo` which is a smaller version of the original YOLO model. It's significantly fast but less accurate.

```python
bbox, label, conf = cv.detect_common_objects(img, confidence=0.25, model='yolov4-tiny')
```
Check out the [example](examples/object_detection_webcam.py) to learn more. 

Other supported models: YOLOv3, YOLOv3-tiny.

### Custom trained YOLO weights
To run inference with custom trained YOLOv3/v4 weights try the following
```python
from cvlib.object_detection import YOLO

yolo = YOLO(weights, config, labels)
bbox, label, conf = yolo.detect_objects(img)
yolo.draw_bbox(img, bbox, label, conf)
```
To enable GPU
```python
bbox, label, conf = yolo.detect_objects(img, enable_gpu=True)
```

Checkout the [example](examples/yolo_custom_weights_inference.py) to learn more.

### Sample output :

![](examples/images/object_detection_output.jpg)

## Utils
### Video to frames
`get_frames( )` method can be helpful when you want to grab all the frames from a video. Just pass the path to the video, it will return all the frames in a list. Each frame in the list is a numpy array.
```python
import cvlib as cv
frames = cv.get_frames('~/Downloads/demo.mp4')
```
Optionally you can pass in a directory path to save all the frames to disk.
```python
frames = cv.get_frames('~/Downloads/demo.mp4', '~/Downloads/demo_frames/')
```

### Creating gif(for less memory isage and faster processing time)
`animate( )` method lets you create gif from a list of images. Just pass a list of images or path to a directory containing images and output gif name as arguments to the method, it will create a gif out of the images and save it to disk for you.

```python
cv.animate(frames, '~/Documents/frames.gif')
```

