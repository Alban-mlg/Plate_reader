"""
Script to verify the installation of machine learning libraries.
"""

import tensorflow as tf
import torch

print(f"TensorFlow version: {tf.__version__}")
print(f"PyTorch version: {torch.__version__}")