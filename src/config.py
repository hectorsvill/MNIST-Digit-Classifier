# src/config.py
"""
Central configuration file
"""
import os

# Data parameters
NUM_CLASSES = 10
INPUT_SHAPE = (28 * 28,) # Flattened 28x28 images

# Training parameters
EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 10000 # Number of samples from training set for validation

# Model saving
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models') # Relative path to models/ dir
MODEL_FILENAME = "mnist_digit_guesser_model.keras"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# API parameters
API_HOST = "0.0.0.0"
API_PORT = 5000
API_LEARNING_RATE_ONLINE = 0.0001 # Smaller LR for online updates

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)