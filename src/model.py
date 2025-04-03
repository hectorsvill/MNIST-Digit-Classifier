# src/model.py
"""
Defines the Keras model architecture.
"""
import tensorflow as tf
from . import config # Use config from the same package

def build_our_cool_model(input_shape=config.INPUT_SHAPE, num_classes=config.NUM_CLASSES):
    """Builds the simple digit-guessing neural network."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape, name="pixels_in"),
        tf.keras.layers.Dense(128, activation='relu', name="thinking_layer_1"),
        tf.keras.layers.Dropout(0.2, name="forget_a_bit_1"),
        tf.keras.layers.Dense(64, activation='relu', name="thinking_layer_2"),
        tf.keras.layers.Dropout(0.2, name="forget_a_bit_2"),
        tf.keras.layers.Dense(num_classes, activation='softmax', name="the_guess_output")
    ], name="mnist_digit_guesser")
    print("Model built successfully.")
    return model