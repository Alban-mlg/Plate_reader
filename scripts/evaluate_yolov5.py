"""
Script to evaluate and visualize YOLOv5 model predictions on the test dataset.

This script loads a trained YOLOv5 model, evaluates its performance on a test dataset,
and visualizes the predictions. It uses environment variables for configuration and
implements error handling and logging.
"""

import os
import torch
import logging
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables for configuration
MODEL_PATH = os.environ.get('YOLOV5_MODEL_PATH', '/home/ubuntu/project/yolov5/runs/train/exp5/weights/best.pt')
TEST_DATA_PATH = os.environ.get('TEST_DATA_PATH', '/home/ubuntu/project/data/test')
OUTPUT_PATH = os.getenv('OUTPUT_PATH', '/home/ubuntu/project/evaluation_results')

def load_model(model_path):
    """
    Load the trained YOLOv5 model.

    Args:
        model_path (str): Path to the trained model file.

    Returns:
        torch.nn.Module: Loaded YOLOv5 model.

    Raises:
        Exception: If there's an error loading the model.
    """
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def evaluate_model(model, test_data_path):
    """
    Evaluate the model on the test dataset.

    Args:
        model (torch.nn.Module): Loaded YOLOv5 model.
        test_data_path (str): Path to the test dataset.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    model.eval()
    results = []
    try:
        for img_path in Path(test_data_path).glob('*'):
            if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                img = Image.open(img_path)
                prediction = model(img)
                results.append(prediction)

        metrics = model.get_metrics(results)
        logging.info(f"Evaluation metrics - mAP: {metrics['mAP']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

        # Save metrics to file
        with open(Path(OUTPUT_PATH) / 'evaluation_metrics.txt', 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

        # Call visualization functions
        visualize_predictions(model, test_data_path, OUTPUT_PATH)
        generate_confusion_matrix(model, test_data_path, OUTPUT_PATH)

        return metrics
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

def visualize_predictions(model, test_data_path, output_path):
    """
    Visualize model predictions on test images.

    Args:
        model (torch.nn.Module): Loaded YOLOv5 model.
        test_data_path (str): Path to the test dataset.
        output_path (str): Path to save the visualizations.
    """
    model.eval()
    try:
        for img_path in Path(test_data_path).glob('*'):
            if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                img = Image.open(img_path)
                prediction = model(img)

                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except IOError:
                    font = ImageFont.load_default()

                for det in prediction.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = det
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    draw.text((x1, y1 - 20), f"License Plate: {conf:.2f}", fill="red", font=font)

                output_file = Path(output_path) / f"pred_{img_path.name}"
                img.save(output_file)
                logging.info(f"Saved prediction visualization: {output_file}")
    except Exception as e:
        logging.error(f"Error during prediction visualization: {e}")
        raise

def generate_confusion_matrix(model, test_data_path, output_path):
    """
    Generate and save the confusion matrix.

    Args:
        model (torch.nn.Module): Loaded YOLOv5 model.
        test_data_path (str): Path to the test dataset.
        output_path (str): Path to save the confusion matrix.
    """
    true_labels = []
    predicted_labels = []
    model.eval()
    try:
        for img_path in Path(test_data_path).glob('*'):
            if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                img = Image.open(img_path)
                prediction = model(img)
                true_label = int(Path(img_path).stem.split('_')[0])  # Assuming filename format: "class_imagename.jpg"
                pred_label = int(prediction.pred[0][:, -1].cpu().numpy()[0])
                true_labels.append(true_label)
                predicted_labels.append(pred_label)

        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(Path(output_path) / 'confusion_matrix.png')
        plt.close()
        logging.info(f"Saved confusion matrix: {Path(output_path) / 'confusion_matrix.png'}")
    except Exception as e:
        logging.error(f"Error generating confusion matrix: {e}")
        raise

def calculate_average_confidence(model, test_data_path):
    correct_confidences = []
    incorrect_confidences = []
    model.eval()
    try:
        for img_path in Path(test_data_path).glob('*'):
            if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                img = Image.open(img_path)
                prediction = model(img)
                true_label = int(Path(img_path).stem.split('_')[0])
                pred_label = int(prediction.pred[0][:, -1].cpu().numpy()[0])
                confidence = float(prediction.pred[0][:, 4].cpu().numpy()[0])

                if true_label == pred_label:
                    correct_confidences.append(confidence)
                else:
                    incorrect_confidences.append(confidence)

        avg_correct_conf = sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0
        avg_incorrect_conf = sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0

        logging.info(f"Average confidence for correct predictions: {avg_correct_conf:.4f}")
        logging.info(f"Average confidence for incorrect predictions: {avg_incorrect_conf:.4f}")

        return avg_correct_conf, avg_incorrect_conf
    except Exception as e:
        logging.error(f"Error calculating average confidence: {e}")
        raise

if __name__ == "__main__":
    try:
        # Ensure output directory exists
        Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

        if not MODEL_PATH:
            raise ValueError("YOLOV5_MODEL_PATH environment variable is not set")

        model = load_model(MODEL_PATH)
        metrics = evaluate_model(model, TEST_DATA_PATH)
        avg_correct_conf, avg_incorrect_conf = calculate_average_confidence(model, TEST_DATA_PATH)
        logging.info("Evaluation, visualization, and confidence calculation completed successfully")
    except Exception as e:
        logging.error(f"An error occurred during script execution: {e}")