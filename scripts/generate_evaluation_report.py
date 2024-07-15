import os
import logging
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RESULTS_DIR = "/home/ubuntu/project/results"
TEMPLATE_PATH = "/home/ubuntu/project/reports/evaluation_report_template.md"
OUTPUT_REPORT_PATH = "/home/ubuntu/project/reports/final_evaluation_report.md"

def read_metrics(metrics_file):
    metrics = {}
    try:
        with open(metrics_file, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                metrics[key] = float(value)
        logging.info(f"Successfully read metrics from {metrics_file}")
        return metrics
    except FileNotFoundError:
        logging.error(f"Metrics file not found: {metrics_file}")
        return {}
    except Exception as e:
        logging.error(f"Error reading metrics file: {metrics_file}. Error: {str(e)}")
        return {}

def load_visualizations():
    visualizations = {}
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(RESULTS_DIR, filename)
                img = Image.open(img_path)
                visualizations[filename] = img
                logging.info(f"Successfully loaded visualization: {filename}")
            except Exception as e:
                logging.error(f"Error loading visualization {filename}: {str(e)}")
    return visualizations

def generate_report_content(metrics, visualizations):
    content = "# Evaluation Report\n\n"

    # Add metrics
    content += "## Model Performance Metrics\n\n"
    key_metrics = ['mAP', 'Precision', 'Recall']
    for metric in key_metrics:
        if metric in metrics:
            content += f"- **{metric}**: {metrics[metric]:.4f}\n"
    content += "\nOther metrics:\n"
    for key, value in metrics.items():
        if key not in key_metrics:
            content += f"- **{key}**: {value:.4f}\n"
    content += "\n"

    # Add visualizations
    content += "## Visualizations\n\n"
    for filename, img in visualizations.items():
        content += f"### {filename}\n\n"
        content += f"![{filename}](../results/{filename})\n\n"

    return content

def update_report_template(template_path, content):
    try:
        with open(template_path, 'r') as f:
            template = f.read()

        # Replace placeholder with actual content
        final_report = template.replace("{{EVALUATION_RESULTS}}", content)

        with open(OUTPUT_REPORT_PATH, 'w') as f:
            f.write(final_report)

        logging.info(f"Successfully generated final report: {OUTPUT_REPORT_PATH}")
    except FileNotFoundError:
        logging.error(f"Template file not found: {template_path}")
    except Exception as e:
        logging.error(f"Error updating report template: {str(e)}")

def main():
    metrics = read_metrics(os.path.join(RESULTS_DIR, "evaluation_metrics.txt"))
    visualizations = load_visualizations()
    report_content = generate_report_content(metrics, visualizations)
    update_report_template(TEMPLATE_PATH, report_content)

if __name__ == "__main__":
    main()