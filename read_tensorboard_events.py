import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import os
import glob

def read_scalar_values(event_file):
    print("Starting to read scalar values from event file")
    scalars = {}
    event_count = 0
    try:
        for event in summary_iterator(event_file):
            event_count += 1
            if event_count % 1000 == 0:
                print(f"Processing event {event_count}...")
            for value in event.summary.value:
                if value.HasField('simple_value'):
                    if value.tag not in scalars:
                        scalars[value.tag] = []
                    scalars[value.tag].append((event.step, value.simple_value))
    except Exception as e:
        print(f"Error reading event file: {str(e)}")
    print(f"Finished processing {event_count} events")
    print(f"Found {len(scalars)} tags with scalar values")
    return scalars

if __name__ == '__main__':
    log_dir = '/home/ubuntu/project/yolov5/runs/train'
    exp_dir = os.path.join(log_dir, 'exp6')

    # Find all event files in the exp_dir
    event_files = glob.glob(os.path.join(exp_dir, 'events.out.tfevents.*'))
    print(f"Found {len(event_files)} event files")

    if not event_files:
        print(f"No TensorBoard event files found in {exp_dir}")
        exit(1)

    # Sort event files by modification time (most recent first)
    event_files.sort(key=os.path.getmtime, reverse=True)

    # Use the most recent event file
    event_file = event_files[0]
    print(f"Selected event file: {event_file}")

    print(f"Reading from {event_file}")
    scalars = read_scalar_values(event_file)
    print(f"Read {len(scalars)} tags from the event file")

    for tag, values in scalars.items():
        print(f"  {tag}: {len(values)} scalar values found")
        if len(values) <= 10:
            print("    All values:")
            for step, value in values:
                print(f"      Step {step}: {value}")
        else:
            print("    First 5 values:")
            for step, value in values[:5]:
                print(f"      Step {step}: {value}")
            print("    Last 5 values:")
            for step, value in values[-5:]:
                print(f"      Step {step}: {value}")
        print()