# Plate Reader

## Description
This project, Plate Reader, is an AI-powered system designed to detect and recognize license plate numbers from images, even under challenging conditions such as blurriness. It utilizes advanced machine learning techniques and the YOLOv5 model to ensure accurate recognition.

## Installation
To set up the Plate Reader project, follow these steps:

1. Set up a Python virtual environment (Python 3.10.12 recommended):
   ```
   python3 -m venv plate_reader_env
   source plate_reader_env/bin/activate
   ```

2. Install dependencies:
   ```
   pip install torch numpy pandas opencv-python tensorflow
   ```

3. Clone the repository:
   ```
   git clone https://github.com/Alban-mlg/Plate_reader.git
   cd Plate_reader
   ```

4. Download the YOLOv5 weights file (will be generated after training)

## Usage
To run the license plate detection:

1. Use the following command:
   ```
   python detect.py --source path/to/image --weights path/to/best.pt
   ```
   Replace `path/to/image` with the path to your input image and `path/to/best.pt` with the path to the trained weights file.

2. Interpreting results:
   - The script will output detected license plates with bounding boxes drawn around them.
   - Each detection will have an associated confidence score.
   - Results will be saved in the `runs/detect` directory.

## Training
To train the model on your own dataset:

1. Prepare your dataset following the YOLOv5 format.
2. Run the training script:
   ```
   python train.py --img 640 --batch 4 --epochs 300 --data license_plates.yaml --weights yolov5s.pt
   ```
   Adjust parameters as needed.

Note: This project uses the License Plates Dataset from Roboflow for training.

## Contributing
We welcome contributions to the Plate Reader project. Please follow these guidelines:

1. Code Style: Adhere to PEP 8 guidelines for Python code.
2. Pull Request Process:
   - Fork the repository
   - Create a feature branch (`git checkout -b feature/AmazingFeature`)
   - Commit your changes (`git commit -m 'Add some AmazingFeature'`)
   - Push to the branch (`git push origin feature/AmazingFeature`)
   - Open a Pull Request
3. Documentation: Please provide documentation for new features or changes.
4. Testing: Ensure adequate test coverage for new features.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
We would like to thank the creators of YOLOv5 and the Roboflow team for providing the License Plates Dataset used in this project.