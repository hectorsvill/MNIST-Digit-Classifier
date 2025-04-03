# src/api/preprocessing.py
import numpy as np
import logging
from .. import config

def preprocess_api_input(features):
    """
    Takes a list/array of pixel values, checks, normalizes (if scale strongly
    suggests 0-255), clamps, and shapes it for model prediction via the API.
    """
    expected_features = config.INPUT_SHAPE[0]
    if not isinstance(features, (list, np.ndarray)):
        raise TypeError("Input 'features' must be a list or NumPy array.")

    # Use float64 for intermediate calculations to potentially reduce precision issues, though float32 is standard for TF
    input_array = np.array(features, dtype=np.float32)

    if input_array.ndim == 0 or input_array.size != expected_features:
         raise ValueError(f"Expected {expected_features} features, but got {input_array.size}")

    # Make a copy to modify
    current_array = input_array.copy()
    original_min = np.min(current_array) # Use min/max of the copy
    original_max = np.max(current_array)

    # --- Heuristic Normalization (High Threshold) ---
    # Only assume 0-255 scale if max value is significantly large AND min is non-negative
    SCALE_THRESHOLD_HIGH = 20.0 # Values <= this threshold will NOT be auto-normalized

    if original_max > SCALE_THRESHOLD_HIGH and original_min >= 0.0:
         logging.debug(
             f"Input max ({original_max}) > {SCALE_THRESHOLD_HIGH} and min >= 0. "
             f"Assuming 0-255 scale and normalizing."
         )
         current_array = current_array / 255.0
    # --- End Heuristic Normalization ---


    # --- Clamping (Always Applied After Potential Normalization) ---
    # Get potentially updated min/max AFTER normalization step
    current_min = np.min(current_array)
    current_max = np.max(current_array)

    # Check if the current array (original or normalized) is outside [0, 1]
    if current_min < 0.0 or current_max > 1.0:
        logging.warning(
            f"Values (after potential normalization) found outside [0, 1] range "
            f"(min: {current_min:.4f}, max: {current_max:.4f}). Clamping to [0, 1]."
            # Log original values for context
            f" Original min/max: {np.min(input_array)}/{np.max(input_array)}"
        )
        current_array = np.clip(current_array, 0.0, 1.0)
    # --- End Clamping ---

    # Reshape for the model: (1, 784)
    return current_array.reshape(1, -1)