# Plate Reader

## Description
Plate Reader is an AI-powered system designed to detect and recognize license plate numbers from images, even under challenging conditions such as blurriness. It utilizes advanced machine learning techniques and the YOLOv5 model to ensure accurate recognition. This project aims to provide a robust solution for automatic license plate detection and recognition.

## Repository Structure
- `datasets/`: Contains the license plate images and annotations used for training and evaluating the model.
- `scripts/`: Includes Python scripts for data preprocessing and model evaluation.
- `yolov5/`: The YOLOv5 model directory with necessary configurations and weights.

## Installation
To set up the project environment:
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Plate_reader.git
   cd Plate_reader
   ```
2. Create a virtual environment (optional but recommended):
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```
   pip install numpy pandas opencv-python tensorflow torch
   ```

## Data Preprocessing
The data preprocessing script (`scripts/data_preprocessing.py`) performs the following steps:
1. Automates YOLO annotations for license plates
2. Resizes and augments images to improve model generalization
3. Splits the dataset into train, validation, and test sets
4. Performs integrity checks on the data to ensure quality

To run the data preprocessing:
```
python scripts/data_preprocessing.py
```

## Usage
1. Training the Model:
   To train the YOLOv5 model on the license plate dataset:
   ```
   python3 train.py --img 640 --batch 2 --epochs 300 --data license_plates.yaml --weights yolov5s.pt --cache
   ```

2. Evaluating the Model:
   After training, evaluate the model's performance:
   ```
   python scripts/evaluate_yolov5.py
   ```

3. Monitoring Training Progress:
   Use TensorBoard to visualize training metrics:
   ```
   tensorboard --logdir runs/train
   ```

## Evaluation
The evaluation script (`scripts/evaluate_yolov5.py`) performs the following:
1. Calculates standard metrics: Precision, Recall, mAP50, mAP50-95
2. Generates a Precision-Recall curve
3. Creates a confusion matrix
4. Calculates average confidence for correct and incorrect predictions
5. Visualizes model predictions on test images

All evaluation results and visualizations are saved in the output directory specified in the script.

## Current Status
As of epoch 3/299, the training metrics are as follows:
- Precision: 0.714
- Recall: 0.562
- mAP50: 0.655
- mAP50-95: 0.463

The training is ongoing, currently at epoch 3 out of 300 planned epochs.

## Next Steps
1. Complete the 300 epochs of training
2. Evaluate the model using the evaluate_yolov5.py script
3. Analyze the precision-recall curve and confusion matrix to understand model performance
4. Review the average confidence metrics for insights into model certainty
5. Analyze results and potentially iterate on the model to improve performance
6. Create comprehensive visualizations and reports to communicate the evaluation results

## Dataset
This project uses the License Plates Dataset from Roboflow, containing 350 images of license plates.

## Environment
- Python 3.10.12
- Key libraries: NumPy, Pandas, OpenCV, TensorFlow, PyTorch

## Contributing
Contributions to the Plate Reader project are welcome. Please ensure to follow these steps:
1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Roboflow for providing the License Plates Dataset
- The YOLOv5 team for their excellent object detection model