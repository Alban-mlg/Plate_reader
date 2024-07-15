# License Plate Recognition AI

## Description
This project implements a license plate recognition system using artificial intelligence. It utilizes the YOLOv5 model to detect and recognize license plates in images and video streams.

## Installation
### Requirements
- Python 3.10.12
- Virtual environment (recommended)

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
1. Preprocess the data:
   ```
   python preprocess_data.py
   ```

2. Train the YOLOv5 model:
   ```
   python train_yolov5.py
   ```

3. Run inference:
   ```
   python test_yolov5_model.py --image path/to/image.jpg
   ```

## Dataset
This project uses the License Plates Dataset from Roboflow. The dataset is structured as follows:
- `train/`: Training images and annotations
- `valid/`: Validation images and annotations
- `test/`: Test images and annotations

## Model
We use the YOLOv5 model for license plate detection and recognition. The trained weights file can be found in the `weights/` directory.

## Evaluation
Evaluation metrics include precision, recall, and mAP (mean Average Precision). To run the evaluation:
```
python evaluate_model.py
```

## Results
(This section will be updated with performance metrics and sample results once the model training and evaluation are complete.)

## Contributing
We welcome contributions to improve the License Plate Recognition AI project. Please follow these steps:
1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License
This project is licensed under the MIT License. See the LICENSE file for details.