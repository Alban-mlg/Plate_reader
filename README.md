# License Plate Detection AI

## Description
This project implements an AI model for detecting and recognizing license plates in images and video streams, even in challenging conditions such as blurriness. It utilizes the YOLOv5 architecture to provide accurate and efficient license plate detection and recognition.

## Dataset
- Uses the License Plates Dataset from Roboflow
- 350 images of license plates
- Data structure: training, validation, and testing sets
- Annotations in YOLO format
- Diverse collection of license plate images from various countries, captured under different lighting conditions and angles

## Model
- YOLOv5 architecture implementation by Ultralytics
- Single-class detection (license plates)
- Trained on CPU (AMD EPYC 7571 with 2 cores)

## Installation
### Requirements
- Python 3.10.12
- Virtual environment (recommended)
- PyTorch and other dependencies (installation instructions provided below)

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/your-username/Plate_reader.git
   cd Plate_reader
   ```

2. Create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required libraries:
   ```
   pip install numpy pandas opencv-python tensorflow torch matplotlib seaborn
   ```

## Usage
### Training
To train the model:
```
python train.py --img 640 --batch 16 --epochs 100 --data license_plates.yaml --weights yolov5s.pt
```

### Inference
To run inference on new images:
```
python detect.py --source path/to/images --weights runs/train/exp/weights/best.pt
```

## Evaluation
Evaluation metrics include precision, recall, and mAP (mean Average Precision). The `evaluate_yolov5.py` script generates these metrics along with visualizations and a detailed report. To run the evaluation:
```
python scripts/evaluate_yolov5.py
```

TensorBoard is used for performance monitoring and visualization.

## Contributing
We welcome contributions to improve the License Plate Detection AI project. Please follow these steps:
1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request using the `gh` CLI:
   ```
   gh pr create --title "Your PR title" --body "Description of your changes"
   ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Roboflow for providing the License Plates Dataset
- Ultralytics for the YOLOv5 implementation
