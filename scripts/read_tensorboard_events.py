# This script will read the TensorBoard event files and extract scalar data using TensorFlow's TFRecordDataset.
import tensorflow as tf
import argparse
import os

# Set up argument parsing
parser = argparse.ArgumentParser(description='Read TensorBoard events and extract scalar data.')
parser.add_argument('--logdir', type=str, required=True, help='Directory containing the TensorBoard event files.')
args = parser.parse_args()

# Function to read and print scalar data from the event files
def read_tensorboard_events(logdir):
    print(f'Reading TensorBoard events from: {logdir}')
    scalar_data = {}
    event_files = [os.path.join(logdir, f) for f in os.listdir(logdir) if 'tfevents' in f]
    for event_file in event_files:
        print(f'Processing file: {event_file}')
        try:
            for event in tf.data.TFRecordDataset([event_file]):
                event = tf.compat.v1.Event.FromString(event.numpy())
                for value in event.summary.value:
                    if value.HasField('simple_value'):
                        if value.tag not in scalar_data:
                            scalar_data[value.tag] = []
                        scalar_data[value.tag].append((event.step, value.simple_value))
                        print(f"Step: {event.step}, Tag: {value.tag}, Value: {value.simple_value}")
        except Exception as e:
            print(f'Error processing file {event_file}: {e}')
    return scalar_data

# Execute the function and print the results
if __name__ == '__main__':
    scalar_data = read_tensorboard_events(args.logdir)
    if scalar_data:
        print("\nSummary of scalar data:")
        for tag, values in scalar_data.items():
            print(f'{tag}:')
            for step, value in values:
                print(f'  Step {step}: Value {value}')
    else:
        print("No scalar data found. Please check if the training has progressed enough to generate scalar data.")