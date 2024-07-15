"""
This script automates the preprocessing of the license plate dataset for YOLOv5 model training.
It includes functions for reading annotations, loading and inspecting images, resizing,
augmenting data, splitting the dataset, and verifying dataset integrity.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import imgaug.augmenters as iaa
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the dataset base path (replace with actual path or use environment variable)
DATASET_BASE_PATH = Path(os.getenv('DATASET_PATH', '/home/ubuntu/project/datasets'))

def read_annotations(annotation_path):
    """
    Reads YOLO format annotations from the specified path.

    Parameters:
    - annotation_path: Path to the annotation file.

    Returns:
    - A list of dictionaries containing annotations.

    Raises:
    - IOError: If there's an issue reading the annotation file.
    """
    try:
        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        annotations = []
        for line_num, line in enumerate(lines, 1):
            values = line.strip().split()
            if len(values) == 5:
                class_id, x_center, y_center, width, height = map(float, values)
                annotations.append({
                    'class_id': int(class_id),
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
            else:
                logging.warning(f"Invalid annotation at line {line_num} in {annotation_path}")

        logging.info(f"Successfully read {len(annotations)} annotations from {annotation_path}")
        return annotations
    except IOError as e:
        logging.error(f"Error reading annotation file {annotation_path}: {str(e)}")
        raise

def load_dataset(dataset_path):
    """
    Load dataset images and corresponding annotations.

    Parameters:
    - dataset_path: Path to the dataset directory.

    Returns:
    - A tuple containing lists of images and annotations.
    """
    images = []
    annotations = []

    try:
        # Iterate over the image files and read corresponding annotations
        for image_path in dataset_path.glob('**/*.jpg'):
            try:
                # Read the image
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Failed to read image: {image_path}")
                images.append(image)

                # Construct the annotation file path
                annotation_file = image_path.with_suffix('.txt')

                # Read the annotations
                image_annotations = read_annotations(annotation_file)
                annotations.append(image_annotations)
            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}")

        logging.info(f"Loaded {len(images)} images and annotations from {dataset_path}")
        return images, annotations
    except Exception as e:
        logging.error(f"Error loading dataset from {dataset_path}: {str(e)}")
        raise

def inspect_dataset(images, annotations):
    """
    Inspect the dataset structure and provide summary statistics.

    Parameters:
    - images: List of image arrays
    - annotations: List of annotation dictionaries

    Returns:
    None
    """
    try:
        logging.info(f"Total number of images: {len(images)}")
        logging.info(f"Total number of annotation files: {len(annotations)}")

        # Check for mismatches between images and annotations
        if len(images) != len(annotations):
            logging.warning("Mismatch between number of images and annotation files")

        # Check image dimensions
        image_sizes = set(img.shape[:2] for img in images)
        logging.info(f"Unique image sizes: {image_sizes}")

        # Check number of annotations per image
        annotations_per_image = [len(ann) for ann in annotations]
        logging.info(f"Min annotations per image: {min(annotations_per_image)}")
        logging.info(f"Max annotations per image: {max(annotations_per_image)}")
        logging.info(f"Average annotations per image: {sum(annotations_per_image) / len(annotations_per_image):.2f}")
    except Exception as e:
        logging.error(f"Error inspecting dataset: {str(e)}")
        raise

def resize_images(images, target_size=(224, 224)):
    """
    Resize images to a consistent size.

    Parameters:
    - images: List of input images.
    - target_size: Tuple of (width, height) for the target size. Default is (224, 224).

    Returns:
    - List of resized images.
    """
    resized_images = []
    try:
        for img in images:
            resized_img = cv2.resize(img, target_size)
            resized_images.append(resized_img)
        logging.info(f"Successfully resized {len(images)} images to {target_size}")
        return resized_images
    except Exception as e:
        logging.error(f"Error resizing images: {str(e)}")
        raise

def augment_data(images, annotations):
    """
    Apply data augmentation to images and annotations.

    Parameters:
    - images: List of input images.
    - annotations: List of corresponding annotations.

    Returns:
    - Tuple of augmented images and annotations.
    """
    try:
        # Define the augmentation pipeline
        seq = iaa.Sequential([
            iaa.Affine(rotate=(-25, 25)),  # Random rotation between -25 and 25 degrees
            iaa.Fliplr(0.5),  # Horizontal flip 50% of the time
            # More augmentations can be added here
        ])

        augmented_images = []
        augmented_annotations = []

        for img, ann in zip(images, annotations):
            # Apply the augmentation pipeline to the images and annotations
            image_aug, annotations_aug = seq(image=img, bounding_boxes=ann)
            augmented_images.append(image_aug)
            augmented_annotations.append(annotations_aug)

        logging.info(f"Successfully augmented {len(images)} images and annotations.")
        return augmented_images, augmented_annotations
    except Exception as e:
        logging.error(f"Error during data augmentation: {str(e)}")
        raise

def split_dataset(images, annotations, test_size=0.2, val_size=0.1):
    """
    Split the dataset into training, validation, and testing sets.

    Parameters:
    - images: List of images to split
    - annotations: List of annotations corresponding to the images
    - test_size: Fraction of the dataset to be used for testing (default: 0.2)
    - val_size: Fraction of the training set to be used for validation (default: 0.1)

    Returns:
    - Tuple containing split images and annotations for training, validation, and testing
    """
    try:
        # Split the dataset into training and testing sets
        images_train, images_test, annotations_train, annotations_test = train_test_split(
            images, annotations, test_size=test_size, random_state=42
        )

        # Split the training set into training and validation sets
        images_train, images_val, annotations_train, annotations_val = train_test_split(
            images_train, annotations_train, test_size=val_size, random_state=42
        )

        logging.info(f"Dataset split: {len(images_train)} training, {len(images_val)} validation, {len(images_test)} testing")
        return images_train, images_val, images_test, annotations_train, annotations_val, annotations_test
    except Exception as e:
        logging.error(f"Error splitting dataset: {str(e)}")
        raise

def verify_dataset_integrity(images, annotations):
    """
    Verify the integrity of the dataset by checking if all images have corresponding annotation files.

    Parameters:
    - images: List of image file paths
    - annotations: List of annotation file paths (not used in this function, but kept for consistency)

    Returns:
    - None
    """
    try:
        missing_annotations = []
        for img_path in images:
            annotation_path = img_path.replace('.jpg', '.txt').replace('images', 'labels')
            if not os.path.exists(annotation_path):
                missing_annotations.append(img_path)

        if missing_annotations:
            logging.warning(f"Missing annotation files for {len(missing_annotations)} images.")
            logging.debug(f"Missing annotation files: {missing_annotations}")
        else:
            logging.info("All images have corresponding annotation files.")
    except Exception as e:
        logging.error(f"Error verifying dataset integrity: {str(e)}")
        raise

# Main script execution
if __name__ == "__main__":
    try:
        logging.info("Starting data preprocessing")

        # Load the training dataset
        train_images, train_annotations = load_dataset(DATASET_BASE_PATH / 'train/images')
        logging.info(f"Loaded {len(train_images)} images and {len(train_annotations)} annotations")

        # Inspect the dataset
        inspect_dataset(train_images, train_annotations)

        # Resize images
        target_size = (224, 224)  # Example target size, can be changed based on model requirements
        train_images_resized = resize_images(train_images, target_size)
        logging.info(f"Resized {len(train_images_resized)} images to {target_size}")

        # Apply data augmentation
        train_images_augmented, train_annotations_augmented = augment_data(train_images_resized, train_annotations)
        logging.info(f"Augmented dataset: {len(train_images_augmented)} images")

        # Split the dataset into training, validation, and testing sets
        train_images, val_images, test_images, train_annotations, val_annotations, test_annotations = split_dataset(
            train_images_augmented, train_annotations_augmented
        )
        logging.info(f"Dataset split: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")

        # Verify the integrity of the dataset
        verify_dataset_integrity(train_images + val_images + test_images, train_annotations + val_annotations + test_annotations)

        logging.info("Data preprocessing completed successfully")

    except Exception as e:
        logging.error(f"An error occurred during data preprocessing: {str(e)}")
        raise

    # TODO: Convert annotations to the required format if necessary