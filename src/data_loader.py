# src/data_loader.py
"""
Handles loading and preprocessing of the MNIST dataset.
"""
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split # Can use this or manual split
from . import config # Use config from the same package

def load_and_preprocess_mnist(validation_split_size=config.VALIDATION_SPLIT, num_classes=config.NUM_CLASSES):
    """
    Loads the MNIST dataset, preprocesses it (normalize, flatten, one-hot encode),
    and splits into train, validation, and test sets.

    Returns:
        tuple: (x_train, y_train_cat), (x_val, y_val_cat), (x_test, y_test_cat), (x_test_orig, y_test_orig)
               Returns processed data and original test data for evaluation.
    """
    print("Loading MNIST dataset...")
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(f"Initial shapes: Train={x_train_full.shape}, Test={x_test.shape}")

    # --- Preprocessing ---
    # 1. Normalize pixel values to [0, 1] and flatten images
    x_train_full_norm = x_train_full.reshape(-1, config.INPUT_SHAPE[0]).astype('float32') / 255.0
    x_test_norm = x_test.reshape(-1, config.INPUT_SHAPE[0]).astype('float32') / 255.0
    print(f"Shapes after flatten & normalize: Train={x_train_full_norm.shape}, Test={x_test_norm.shape}")

    # 2. One-Hot Encode labels
    y_train_full_cat = tf.keras.utils.to_categorical(y_train_full, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
    print(f"Shapes after one-hot encoding: Train Labels={y_train_full_cat.shape}, Test Labels={y_test_cat.shape}")

    # --- Splitting ---
    # Split the full training data into training and validation sets
    if validation_split_size >= len(x_train_full_norm):
        raise ValueError("Validation split size is too large!")

    # Manual split (alternative: use sklearn.model_selection.train_test_split)
    x_val_norm = x_train_full_norm[-validation_split_size:]
    y_val_cat = y_train_full_cat[-validation_split_size:]
    x_train_norm = x_train_full_norm[:-validation_split_size]
    y_train_cat = y_train_full_cat[:-validation_split_size]

    print(f"Final shapes: Train={x_train_norm.shape}, Validation={x_val_norm.shape}, Test={x_test_norm.shape}")

    # Return original test labels as well for easier evaluation reporting
    return (x_train_norm, y_train_cat), (x_val_norm, y_val_cat), (x_test_norm, y_test_cat), (x_test, y_test)


def create_tf_dataset(features, labels, batch_size=config.BATCH_SIZE, shuffle=False, buffer_size=10000):
    """Creates a tf.data.Dataset for efficient training/evaluation."""
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset