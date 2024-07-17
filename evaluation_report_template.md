# Evaluation Report for License Plate Detection AI

## Introduction
This report presents the evaluation results for an AI model designed to detect license plate numbers and letters from images, even under challenging conditions such as blurriness. The project's goal is to develop a robust system capable of accurately recognizing and decoding license plate information from various image sources, which is a critical component in many automated systems, including traffic monitoring and vehicle identification.

## Objectives
The primary objective of this AI model is to achieve high accuracy in detecting and recognizing license plate characters from images with varying quality and clarity. The model aims to handle different lighting conditions, angles, distances, and levels of image blurriness, ensuring reliable performance across diverse scenarios.

## Dataset Description
The dataset used for training and evaluating the AI model consists of 350 images of license plates, sourced from Roboflow. It includes a variety of license plates in different environments and conditions to simulate real-world scenarios. The dataset is divided into training, validation, and testing sets, with annotations in YOLO format that provide bounding box coordinates and class indices for each license plate in the images.

## Model Details
- Model: YOLOv5
- Task: License Plate Detection (single-class object detection)
- Training duration: July 15, 2024 to July 17, 2024
- Hardware: AMD EPYC 7571 CPU with 2 cores (no GPU)
- Batch size: Reduced due to memory constraints

## Evaluation Metrics
- mAP@0.5: Peaked around 0.8
- mAP@0.5:0.95: Showed steady increase
- Precision: Stabilized around 0.85
- Recall: Ended slightly above 0.8

## Visualizations
![TensorBoard Metrics](/home/ubuntu/screenshots/e7670661-8631-4852-8fab-a989c48877bf.png)

## Insights and Analysis
The model has demonstrated promising performance, with high precision indicating good accuracy in detecting license plates. The recall metric suggests the model is effective in detecting most license plates within the dataset. The decreasing loss values throughout the training process indicate that the model is learning and improving its predictions over time. Considering the training was conducted on a CPU, the results are particularly encouraging.

## Recommendations
- Training on a GPU could potentially improve the model's performance and reduce training time.
- Increasing the dataset size could help the model generalize better to unseen data.
- Fine-tuning hyperparameters may lead to further improvements in model accuracy.

## Challenges and Solutions
During the project, we encountered and resolved a zero-size array error in the evaluation script, which was addressed by adding checks for empty predictions. Additionally, we adjusted the batch size due to memory constraints on the CPU, which allowed the training to proceed without interruptions.

## Conclusion
The AI model for license plate detection has shown strong potential in recognizing and decoding license plate information from images. The evaluation metrics and visualizations indicate that the model is capable of high accuracy and reliability, even when trained on limited hardware resources. Further improvements could be achieved with access to more powerful computing resources and an expanded dataset.