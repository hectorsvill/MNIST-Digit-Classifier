# src/api/app.py
"""
Flask API definition. Now accepts image uploads.
"""
import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
import logging
import time

# Use relative imports within the 'src' package
from .. import config
# Import the new image processing function
from .preprocessing import process_uploaded_image

# --- Global Variables ---
model = None
optimizer = None
loss_fn = tf.keras.losses.CategoricalCrossentropy() # Define loss globally

# --- Load The Model ---
def load_trained_model(model_path=config.MODEL_SAVE_PATH):
    """Loads the trained Keras model and compiles it for potential learning."""
    global model, optimizer # Modify global variables
    try:
        if not os.path.exists(model_path):
            logging.error(f"FATAL: Model file not found at {model_path}. Cannot start API.")
            return False
        model = tf.keras.models.load_model(model_path)
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.API_LEARNING_RATE_ONLINE)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        logging.info(f"Model loaded successfully from {model_path} and compiled for learning.")
        # Optional "warm-up" prediction with dummy data matching expected input shape
        dummy_input = np.zeros((1, config.INPUT_SHAPE[0]), dtype=np.float32)
        _ = model.predict(dummy_input, verbose=0)
        logging.info("Model warmed up.")
        return True
    except Exception as e:
        logging.error(f"FATAL: Error loading model: {e}", exc_info=True)
        return False

# --- Create Flask App ---
def create_app():
    """Factory function to create the Flask app."""
    flask_app = Flask(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not load_trained_model():
        logging.warning("Model loading failed. API endpoints might not work correctly.")

    # --- API Endpoints ---
    @flask_app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        if model:
            return jsonify({"status": "ok", "message": "Model is loaded."}), 200
        else:
            return jsonify({"status": "error", "message": "Model is not loaded!"}), 503

    @flask_app.route('/predict', methods=['POST'])
    def predict():
        """Endpoint for getting predictions from an uploaded image file."""
        if model is None:
            return jsonify({"error": "Model not available."}), 503

        start_time = time.time()

        # Check if the post request has the file part
        if 'image' not in request.files:
            logging.warning("Predict request missing 'image' file part.")
            return jsonify({"error": "No image file part in the request"}), 400

        file = request.files['image']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            logging.warning("Predict request received empty filename.")
            return jsonify({"error": "No selected image file"}), 400

        if file:
            try:
                # 1. Process the uploaded image file
                features = process_uploaded_image(file) # Returns flattened (784,) array

                # 2. Reshape for the model (expects batch dimension)
                processed_input = features.reshape(1, -1) # Shape (1, 784)

                # 3. Predict
                predictions = model.predict(processed_input, verbose=0) # Output shape (1, 10)
                predicted_class = int(np.argmax(predictions[0]))
                confidence = float(np.max(predictions[0]))

                end_time = time.time()
                logging.info(f"Prediction request processed in {end_time - start_time:.4f}s. Result: {predicted_class}")

                return jsonify({
                    "prediction": predicted_class,
                    "confidence": confidence
                })

            except (ValueError, IOError) as ve:
                logging.warning(f"Image processing error: {ve}")
                return jsonify({"error": f"Invalid image or processing error: {ve}"}), 400
            except Exception as e:
                logging.error(f"Prediction error: {e}", exc_info=True)
                return jsonify({"error": "Prediction failed due to an internal error."}), 500
        else:
             # This case should be caught by filename check, but defensively handle
             return jsonify({"error": "Unknown error handling file upload"}), 500


    @flask_app.route('/learn', methods=['POST'])
    def learn():
        """Endpoint for online learning from an uploaded image file and label."""
        if model is None or optimizer is None:
             return jsonify({"error": "Model or optimizer not available for learning."}), 503

        start_time = time.time()

        # Check for image file part
        if 'image' not in request.files:
            logging.warning("Learn request missing 'image' file part.")
            return jsonify({"error": "No image file part in the request"}), 400

        file = request.files['image']
        if file.filename == '':
            logging.warning("Learn request received empty filename.")
            return jsonify({"error": "No selected image file"}), 400

        # Check for label form field
        if 'label' not in request.form:
            logging.warning("Learn request missing 'label' form field.")
            return jsonify({"error": "Missing 'label' form field"}), 400

        label_str = request.form['label']

        if file:
            try:
                # 1. Validate Label
                if not label_str.isdigit():
                     raise ValueError("Label must be an integer digit.")
                label_int = int(label_str)
                if not (0 <= label_int < config.NUM_CLASSES):
                    raise ValueError(f"Label must be an integer between 0 and {config.NUM_CLASSES-1}")

                # 2. Process the uploaded image file
                features = process_uploaded_image(file) # Returns flattened (784,) array

                # 3. Reshape features for the model
                processed_input = features.reshape(1, -1) # Shape (1, 784)

                # 4. One-hot encode the label
                label_categorical = tf.keras.utils.to_categorical([label_int], num_classes=config.NUM_CLASSES) # Shape (1, 10)

                # 5. Perform one learning step
                metrics = model.train_on_batch(processed_input, label_categorical, return_dict=True)

                end_time = time.time()
                logging.info(f"Learn request processed in {end_time - start_time:.4f}s. Label: {label_int}, Loss: {metrics.get('loss', 'N/A'):.4f}, Acc: {metrics.get('accuracy', 'N/A'):.4f}")

                # Convert potential NumPy types before returning JSON
                loss_value = metrics.get('loss')
                accuracy_value = metrics.get('accuracy')
                if isinstance(loss_value, np.generic): loss_value = loss_value.item()
                if isinstance(accuracy_value, np.generic): accuracy_value = accuracy_value.item()
                loss_value = float(loss_value) if loss_value is not None else None
                accuracy_value = float(accuracy_value) if accuracy_value is not None else None

                return jsonify({
                    "message": "Model updated with the provided example.",
                    "label_provided": label_int,
                    "loss_on_example": loss_value,
                    "accuracy_on_example": accuracy_value
                })

            except (ValueError, IOError) as ve:
                logging.warning(f"Image processing or label validation error: {ve}")
                return jsonify({"error": f"Invalid input or processing error: {ve}"}), 400
            except Exception as e:
                logging.error(f"Learning step error: {e}", exc_info=True)
                return jsonify({"error": "Learning step failed due to an internal error."}), 500
        else:
             return jsonify({"error": "Unknown error handling file upload"}), 500

    return flask_app

# Use run_api.py to start the server directly.