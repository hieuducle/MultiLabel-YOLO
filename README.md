# [PYTORCH] Object Detection & Classification (with two header) using YOLO and CNN
<p align="center">
 <h1 align="center">MultiLabel-YOLO</h1>
</p>

## Introduction
This project integrates object detection (YOLO) and image classification (CNN) to identify and classify objects in videos. 
The system first detects objects using YOLO, crops the detected regions, and then classifies them based on their type and color using a custom-trained CNN model.
## Descriptions
* Object Detection: finetune yolov8n
* Classification: A ResNet50-based CNN model <br/>
<p align="center"> <img src="cnn.png" width=350 height=350 ><br/> </p>
* End-to-End Pipeline: Combines detection and classification into a single workflow.
<p align="center">
  <img src="output/output.gif" width=600><br/>
  <i>Camera app demo</i>
</p>


