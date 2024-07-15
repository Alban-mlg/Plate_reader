"""
Isolated testing loop for YOLOv5 model.
"""

import torch
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YOLOv5 model path (update this once weights are available)
MODEL_PATH = "/home/ubuntu/project/yolov5/runs/train/exp/weights/best.pt"

# Directory containing test images
TEST_IMAGES_DIR = "/home/ubuntu/project/data/test"

def load_model(model_path: str):
    """
    Load the YOLOv5 model from the specified path.
    """
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess the input image for YOLOv5.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 640))
    return image

def run_inference(model, image: np.ndarray):
    """
    Run inference on the input image using the YOLOv5 model.
    """
    results = model(image)
    return results

def test_model_on_images(model, image_paths: List[str]):
    """
    Test the YOLOv5 model on a list of image paths.
    """
    for image_path in image_paths:
        image = preprocess_image(image_path)
        results = run_inference(model, image)
        logger.info(f"Results for {image_path}: {results}")

def main():
    """
    Main function to run the isolated testing loop.
    """
    model = load_model(MODEL_PATH)
    if model is None:
        logger.error("Model could not be loaded. Exiting.")
        return

    test_image_paths = list(Path(TEST_IMAGES_DIR).glob('*.jpg'))
    test_model_on_images(model, test_image_paths)

if __name__ == "__main__":
    main()