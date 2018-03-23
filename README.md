# Hedgehog Detector

This project implements a custom object detection model with TensorFlow's object detection API. Images are from Google. There are 200 total images.

This exploratory object detector trained for 1000 iterations on CPU with dataset split at:

- Train images: 160
- Val images: 40

# Requirements

- Python 3+
- TensorFlow 1.4
- OpenCV 3.3
- Jupyter Notebook

# Data collection

I used Chrome extension Fatkun Batch Download to collect 200 images for the data set. I used LabelImg to create bounding box annotations for each image.

Train and val images are saved in separate directories. I created the convert_to_tfrecords.py script and ran it on each directory to create train.tfrecord and val.tfrecord files.

# Project hierarchy

+ models
   + research
      + object_detection
         + out_graph
             - saved_model.pb
         - object_detection_tutorial.ipynb
+ train_object_detector
   - Images
   - Labels
   - convert_to_tfrecords.py
