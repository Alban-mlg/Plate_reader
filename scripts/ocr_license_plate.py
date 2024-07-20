"""
OCR License Plate Recognition Script
This script uses Tesseract OCR to recognize text from license plate images.
"""

import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the path for Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def preprocess_image(image):
    """
    Preprocess the image for OCR.
    """
    try:
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(thresh)

        return contrast
    except Exception as e:
        logging.error(f"Error in image preprocessing: {str(e)}")
        return None

def perform_ocr(image):
    """
    Perform OCR on the preprocessed image.
    """
    try:
        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        if preprocessed_image is None:
            return None, 0

        # Tesseract configuration for license plates
        config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

        # Perform OCR
        ocr_result = pytesseract.image_to_string(preprocessed_image, config=config)

        # Get confidence scores
        ocr_data = pytesseract.image_to_data(preprocessed_image, config=config, output_type=Output.DICT)
        confidences = ocr_data['conf']

        # Calculate average confidence for non-negative values
        valid_confidences = [conf for conf in confidences if conf != -1]
        avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0

        return ocr_result.strip(), avg_confidence
    except Exception as e:
        logging.error(f"Error in OCR process: {str(e)}")
        return None, 0

def debug_output(image, ocr_result, confidence):
    """
    Create a debug image with OCR results overlaid.
    """
    try:
        debug_image = image.copy()

        # Add OCR result and confidence as text on the image
        cv2.putText(debug_image, f"OCR: {ocr_result}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_image, f"Confidence: {confidence:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return debug_image
    except Exception as e:
        logging.error(f"Error in creating debug output: {str(e)}")
        return None

def process_license_plate(image_path):
    """
    Process a license plate image and return OCR results.
    """
    try:
        # Load the image
        image = cv2.imread(image_path)

        if image is None:
            logging.error(f"Failed to load image: {image_path}")
            return

        # Perform OCR
        ocr_result, confidence = perform_ocr(image)

        if ocr_result is None:
            logging.error("OCR process failed")
            return

        # Log results
        logging.info(f"OCR Result: {ocr_result}")
        logging.info(f"Confidence: {confidence:.2f}%")

        # Create debug output
        debug_image = debug_output(image, ocr_result, confidence)

        if debug_image is not None:
            # Save debug image
            debug_path = image_path.replace('.jpg', '_debug.jpg')
            cv2.imwrite(debug_path, debug_image)
            logging.info(f"Debug image saved: {debug_path}")

        return ocr_result, confidence
    except Exception as e:
        logging.error(f"Error in processing license plate: {str(e)}")
        return None, 0

# Main function to run the script
if __name__ == "__main__":
    # Update the image path to the actual path of the downloaded license plate image
    image_path = "/home/ubuntu/project/datasets/license_plates/images/test/uk-license-plate.jpg"
    result, conf = process_license_plate(image_path)

    if result:
        print(f"License Plate: {result}")
        print(f"Confidence: {conf:.2f}%")
    else:
        print("Failed to process license plate")