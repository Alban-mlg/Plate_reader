# Evaluation Report for License Plate Recognition AI

## Introduction
This report presents the evaluation results of an AI model developed for recognizing license plate numbers and letters from images, even under challenging conditions such as blurriness. The project's primary goal was to create a robust system capable of accurately detecting and reading license plates in various real-world scenarios.

Key aspects of the project include:
- Dataset: We utilized the License Plates Object Detection Dataset from Roboflow, comprising 350 images of license plates.
- Model: We employed YOLOv5, a state-of-the-art object detection model, for both detecting and recognizing license plates.

## Methodology

### 1. Data Preprocessing
We implemented an automated data preprocessing pipeline using the `data_preprocessing.py` script. This process included:
- Reading YOLO format annotations
- Loading dataset images and corresponding annotations
- Inspecting the dataset for quality and consistency
- Resizing images to a uniform size for model input
- Applying data augmentation techniques to increase dataset diversity and improve model generalization

### 2. Model Training
We trained the YOLOv5 model using the following command:
```
python3 train.py --img 640 --batch 4 --epochs 300 --data license_plates.yaml --weights yolov5s.pt --cache
```
Due to hardware limitations, the training was performed on CPU. This approach, while slower, allowed us to proceed with model development and testing.

### 3. Evaluation Process
For model evaluation, we developed and utilized the `evaluate_yolov5.py` script. The evaluation process includes:
- Calculating key performance metrics:
  - Mean Average Precision (mAP)
  - Precision
  - Recall
- Generating a confusion matrix for visual performance analysis
- Computing average confidence scores for both correct and incorrect predictions

### 4. Visualization
To provide a comprehensive understanding of the model's performance, we included:
- Sample predictions on test images to showcase the model's capabilities
- A confusion matrix visualization to highlight the model's strengths and areas for improvement

## Model Performance Metrics
- Mean Average Precision (mAP):
- Precision:
- Recall:

## Visualizations
- Predictions on test images:
- Confusion matrix:

## Insights and Recommendations
- Summary of the model's strengths and weaknesses:
- Suggestions for model improvement:

## Conclusion
Final thoughts and next steps.