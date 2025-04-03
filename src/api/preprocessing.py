# src/api/preprocessing.py
import numpy as np
import logging
from .. import config
from PIL import Image, UnidentifiedImageError # Import Pillow
import io # To read image bytes

logger = logging.getLogger(__name__)

def process_uploaded_image(file_storage) -> np.ndarray:
    """
    Reads an uploaded image file, converts to 28x28 grayscale, normalizes,
    and flattens it into a feature vector.

    Args:
        file_storage: A Werkzeug FileStorage object (from request.files).

    Returns:
        np.ndarray: A flattened numpy array of 784 normalized pixel values (float32).

    Raises:
        ValueError: If the image cannot be processed or is invalid.
        IOError: If there's an issue reading the file stream.
    """
    target_size = (28, 28)
    expected_features = target_size[0] * target_size[1] # Should be 784

    try:
        # Read image bytes from the FileStorage stream
        img_bytes = file_storage.read()
        if not img_bytes:
            raise ValueError("Uploaded file is empty.")

        # Open image using Pillow
        img = Image.open(io.BytesIO(img_bytes))

        # 1. Convert to grayscale ('L' mode)
        img_gray = img.convert('L')

        # 2. Resize to target size (e.g., 28x28)
        # Use LANCZOS for potentially better quality downsampling
        img_resized = img_gray.resize(target_size, Image.Resampling.LANCZOS)

        # 3. Convert to NumPy array
        img_array = np.array(img_resized, dtype=np.float32)

        # 4. Normalize pixel values to [0, 1]
        # MNIST models usually expect black=0, white=1 or vice-versa.
        # Often, datasets have white background (255) and black digit (0).
        # Normalizing (pixel / 255.0) makes background 1.0, digit 0.0.
        # Some models are trained expecting the inverse (black background).
        # If predictions are poor, try inverting: normalized_array = 1.0 - (img_array / 255.0)
        normalized_array = img_array / 255.0

        # 5. Flatten the array
        flattened_array = normalized_array.flatten()

        if flattened_array.size != expected_features:
             # This shouldn't happen if resize worked correctly
             raise ValueError(f"Processed image has {flattened_array.size} features, expected {expected_features}.")

        logger.debug(f"Image processed successfully. Shape: {flattened_array.shape}")
        return flattened_array

    except UnidentifiedImageError:
        logger.warning("Cannot identify image file. Is it a valid image format (PNG, JPG, etc.)?")
        raise ValueError("Invalid image format. Please upload a valid PNG, JPG, etc.")
    except IOError as e:
        logger.error(f"IOError reading image stream: {e}")
        raise IOError(f"Could not read image file stream: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during image processing: {e}")
        raise ValueError(f"Failed to process image: {e}")


# --- (Optional) Keep the old function if needed ---
def preprocess_api_input(features):
    """
    Takes a list/array of pixel values, checks, normalizes, and shapes it
    for model prediction via the API. (Used when input is features list)
    """
    # ... (previous implementation of this function) ...
    expected_features = config.INPUT_SHAPE[0]
    if not isinstance(features, (list, np.ndarray)):
        raise TypeError("Input 'features' must be a list or NumPy array.")

    input_array = np.array(features, dtype=np.float32)

    if input_array.ndim == 0 or input_array.size != expected_features:
         raise ValueError(f"Expected {expected_features} features, but got {input_array.size}")

    original_min = np.min(input_array)
    original_max = np.max(input_array)
    current_array = input_array.copy()

    SCALE_THRESHOLD_HIGH = 20.0
    if original_max > SCALE_THRESHOLD_HIGH and original_min >= 0.0:
         logging.debug(f"Input max > {SCALE_THRESHOLD_HIGH}, assuming 0-255 scale and normalizing.")
         current_array = current_array / 255.0

    current_min = np.min(current_array)
    current_max = np.max(current_array)
    if current_min < 0.0 or current_max > 1.0:
        logging.warning(f"Values found outside [0, 1] range. Clamping.")
        current_array = np.clip(current_array, 0.0, 1.0)

    return current_array.reshape(1, -1) # Reshape for model here if needed by caller


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