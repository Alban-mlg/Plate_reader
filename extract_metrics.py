# Import TensorFlow and other necessary libraries
import tensorflow as tf
import os

# Function to extract metrics from TensorBoard logs
def extract_metrics(logdir):
    metrics = {}
    for root, dirs, files in os.walk(logdir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_file = os.path.join(root, file)
                try:
                    for e in tf.compat.v1.train.summary_iterator(event_file):
                        for v in e.summary.value:
                            if v.tag not in metrics:
                                metrics[v.tag] = []
                            metrics[v.tag].append(v.simple_value)
                except tf.errors.DataLossError:
                    print(f"DataLossError in file: {event_file}")
                    continue
    return metrics

# Extract metrics from the 'runs/train' directory
metrics = extract_metrics('runs/train')

# Print the extracted metrics
for tag, values in metrics.items():
    print(f"{tag}: {values[-1]}")  # Print the last value for each metric