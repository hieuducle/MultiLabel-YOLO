# [PYTORCH] Object Detection & Classification (with two header) using YOLO and CNN
<p align="center">
 <h1 align="center">MultiLabel-YOLO</h1>
</p>

## Introduction
This project integrates object detection (YOLO) and image classification (CNN) to identify and classify objects in videos. 
The system first detects objects using YOLO, crops the detected regions, and then classifies them based on their type and color using a custom-trained CNN model.
## Descriptions
* Collected and prepared a custom dataset by recording videos of individual products and extracting image frames.

* Trained a dual-output classification model with ResNet50 backbone to predict both product name and color.

* Fine-tuned YOLOv8 for object detection to locate items in video frames, then passed the cropped regions into the classifier.

* Integrated classification outputs with real-time video processing to display both product type and color.

* Used Python, PyTorch, and OpenCV for end-to-end implementation and visualization.
</br>
<p align="center"> <img src="cnn.png" width=350 height=350 ><br/> </p>
* End-to-End Pipeline: Combines detection and classification into a single workflow.
<p align="center">
  <img src="output/output.gif" width=600><br/>
  <i>Demo</i>
</p>


