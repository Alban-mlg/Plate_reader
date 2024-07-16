# This script will read the TensorBoard event file and extract scalar data using TensorFlow's TFRecordDataset.
import tensorflow as tf
import os

# Path to the TensorBoard event file
event_file_path = '/home/ubuntu/project/yolov5/runs/train/exp6/events.out.tfevents.1721079306.ip-10-240-175-211.215672.0'

# Function to read and print scalar data from the event file
def read_tensorboard_events(event_file_path):
    print(f'Reading TensorBoard events from: {event_file_path}')
    scalar_data = {}
    for event in tf.data.TFRecordDataset([event_file_path]):
        event = tf.compat.v1.Event.FromString(event.numpy())
        for value in event.summary.value:
            if value.HasField('simple_value'):
                if value.tag not in scalar_data:
                    scalar_data[value.tag] = []
                scalar_data[value.tag].append((event.step, value.simple_value))
                print(f"Step: {event.step}, Tag: {value.tag}, Value: {value.simple_value}")
    return scalar_data

# Execute the function and print the results
if __name__ == '__main__':
    scalar_data = read_tensorboard_events(event_file_path)
    print("\nSummary of scalar data:")
    for tag, values in scalar_data.items():
        print(f'{tag}:')
        for step, value in values:
            print(f'  Step {step}: Value {value}')