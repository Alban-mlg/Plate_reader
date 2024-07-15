import torch
import cv2
import numpy as np
import os
import random
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(weights_path):
    # Load YOLOv5 model
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def run_inference(model, image_path):
    # Run inference on a single image
    try:
        img = cv2.imread(image_path)
        results = model(img)
        return results
    except Exception as e:
        logger.error(f"Error during inference on {image_path}: {e}")
        raise

def save_results(results, output_dir, image_name):
    # Save annotated image and text file with detections
    try:
        results.save(save_dir=output_dir)

        # Save text file with detections
        txt_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
        with open(txt_path, 'w') as f:
            for *xyxy, conf, cls in results.xyxy[0]:
                f.write(f"{int(cls)} {conf:.4f} {xyxy[0]:.0f} {xyxy[1]:.0f} {xyxy[2]:.0f} {xyxy[3]:.0f}\n")
    except Exception as e:
        logger.error(f"Error saving results for {image_name}: {e}")
        raise

def create_test_subset(test_dir, subset_dir, num_images=10):
    os.makedirs(subset_dir, exist_ok=True)
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    subset = random.sample(test_images, min(num_images, len(test_images)))
    for img in subset:
        shutil.copy(os.path.join(test_dir, img), subset_dir)
    logger.info(f"Created test subset with {len(subset)} images in {subset_dir}")

def calculate_metrics(results):
    # This is a placeholder. Implement actual metric calculation based on your needs.
    return {"confidence": results.xyxy[0][:, 4].mean().item()}

def main():
    model_path = '/home/ubuntu/project/runs/train/exp/weights/best.pt'
    test_images_dir = '/home/ubuntu/project/data/test'
    subset_dir = '/home/ubuntu/project/data/test_subset'
    output_dir = '/home/ubuntu/project/results'

    os.makedirs(output_dir, exist_ok=True)
    create_test_subset(test_images_dir, subset_dir)

    try:
        model = load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    for image_file in os.listdir(subset_dir):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(subset_dir, image_file)
            try:
                results = run_inference(model, image_path)
                save_results(results, output_dir, image_file)
                metrics = calculate_metrics(results)
                logger.info(f"Processed {image_file}. Metrics: {metrics}")
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")

    logger.info("Testing completed.")

if __name__ == '__main__':
    main()