## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov3/blob/master/requirements.txt) dependencies installed, including `torch>=1.7`. To install run:
```bash
$ pip install -r requirements.txt
```

## Overview
The main obejctive is to find a way to locate the center of a cell phone in an image. The approach I chose was 
to use the object detection algorithm YOLO which locates and detects objects in images. It outputs bounding boxes 
around the identified object which I then use to calculate the normalized coordinates of the center of the cellphone.
A high level explanation of the algorithm can be found here: [link](https://thebinarynotes.com/yolo-realtime-object-detection/) 

I used [ultralytics](https://github.com/ultralytics/yolov3) pytorch implementation as the base of my code. Their model 
is pretrained with the ImageNet dataset which contains cell phone as one of their classes. Due to time constraints
I chose to use their pretrained model rather than training my own class. Some considerations when training the class
would be to use data augmentation tricks such as random crops, rotations, and shifts to make up for the small dataset. 
In order to train with the provided dataset, I would first have to generate the corresponding bounding boxes given the 
object centers. This could be achieved by looking at the area around the center to find the opposing screen corner locations
by considering the the pixel value or using SIFT. While testing with the labeled 
dataset, I noticed that the images where localization was unsuccessful had a small size-of-phone to background ratio. 
I also noticed that the more parallel the floor is to the camera, the better the results. This leads me to believe that
the training data used for the cell phone class didn't contain many images of cell phones from the more extreme points 
of view. Hence, another data augmentation trick we could use to improve the cellphone class is projective transformations. 
We could generate random homography matrices within a certain range to generate more images with these extreme perspectives. 
Alongside the other data augmentation methods mentioned, this has the potential to improve the accuracy of localization. 
Another solution to improve accuracy would be an intermediate step that partitions images where nothing was detected into 
4 distinct images which are then run through the network again. A caveat here would be to correctly upsample the images 
in order for the dimensions to work properly. Generality is important in learning applications but given prior information
on the distribution of images we are to expect, we could apply some image processing techniques that could improve the 
accuracy but lose generality. For example, the phone is facing up and turned off in every image, and in most images the 
phone screen is the darkest part of the image. If this were the expected case, we could mask the image to narrow down the
observed area for example. 

## Results
Achieved 88.4% accuracy on a 0.05 normalized-radius-error threshold in the provided dataset.


## Citation

Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

Redmon, Joseph, and Ali Farhadi. "YOLO9000: better, faster, stronger." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018).



## Run
Inference on a single image or a directory with multiple images
```bash
$ python find_phone.py path/to/dir/or/image
```
Test accuracy on labeled dataset
```bash
$ python find_phone.py path/to/dir/or/image --test --labels_path path/to/labels/file
```
