import os
import tensorflow as tf

def extract_scalar_data(event_dir):
    scalar_data = {}
    print(f"Looking for event files in: {event_dir}")
    for event_file in os.listdir(event_dir):
        print(f"Found file: {event_file}")
        if 'events.out.tfevents' in event_file:
            event_path = os.path.join(event_dir, event_file)
            print(f"Reading event file: {event_path}")
            for e in tf.compat.v1.train.summary_iterator(event_path):
                print(f"Processing event: Step {e.step}, Wall time: {e.wall_time}")
                for v in e.summary.value:
                    print(f"  Processing value: Tag: {v.tag}")
                    if v.HasField('simple_value'):
                        if v.tag not in scalar_data:
                            scalar_data[v.tag] = []
                        scalar_data[v.tag].append((e.step, v.simple_value))
                        print(f"    Extracted scalar data: Tag: {v.tag}, Step: {e.step}, Value: {v.simple_value}")
    if not scalar_data:
        print("No scalar data found.")
    return scalar_data

log_dir = '/home/ubuntu/project/yolov5/runs/train/exp6'

print(f'Reading TensorBoard events from: {log_dir}')
scalar_data = extract_scalar_data(log_dir)

print(f"\nTotal number of scalar data tags found: {len(scalar_data)}")

for tag, values in scalar_data.items():
    print(f'\n{tag}:')
    for step, value in values:
        print(f'  Step {step}: {value}')