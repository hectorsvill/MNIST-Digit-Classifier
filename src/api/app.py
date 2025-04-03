# src/api/app.py
"""
Flask API definition.
"""
import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
import logging
import time

# Use relative imports within the 'src' package
from .. import config
from .preprocessing import preprocess_api_input

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

        # Recreate the optimizer used during training (or a new one for online)
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.API_LEARNING_RATE_ONLINE)

        # Compile the model for train_on_batch
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

        logging.info(f"Model loaded successfully from {model_path} and compiled for learning.")

        # Optional "warm-up" prediction
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

    # Load the model when the app is created
    if not load_trained_model():
        logging.warning("Model loading failed. API endpoints might not work correctly.")
        # You might choose to exit here or let it run but return errors

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
        """Endpoint for getting predictions."""
        if model is None:
            return jsonify({"error": "Model not available."}), 503

        start_time = time.time()
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        if 'features' not in data:
            return jsonify({"error": "Missing 'features' key in JSON input"}), 400

        try:
            processed_input = preprocess_api_input(data['features'])
            predictions = model.predict(processed_input, verbose=0) # Output shape (1, 10)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))

            end_time = time.time()
            logging.info(f"Prediction request processed in {end_time - start_time:.4f}s. Result: {predicted_class}")

            return jsonify({
                "prediction": predicted_class,
                "confidence": confidence
            })

        except (ValueError, TypeError) as ve:
            logging.warning(f"Bad prediction request data: {ve}")
            return jsonify({"error": f"Invalid input data: {ve}"}), 400
        except Exception as e:
            logging.error(f"Prediction error: {e}", exc_info=True)
            return jsonify({"error": "Prediction failed."}), 500

    @flask_app.route('/learn', methods=['POST'])
    def learn():
        """Endpoint for online learning (fine-tuning)."""
        if model is None or optimizer is None: # Check optimizer too
             return jsonify({"error": "Model or optimizer not available for learning."}), 503

        start_time = time.time()
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        if 'features' not in data or 'label' not in data:
            return jsonify({"error": "Missing 'features' or 'label' key"}), 400

        try:
            processed_input = preprocess_api_input(data['features']) # Shape (1, 784)
            label = data['label']

            if not isinstance(label, int) or not (0 <= label < config.NUM_CLASSES):
                raise ValueError(f"Label must be an integer between 0 and {config.NUM_CLASSES-1}")

            label_categorical = tf.keras.utils.to_categorical([label], num_classes=config.NUM_CLASSES) # Shape (1, 10)

            # Perform one learning step
            metrics = model.train_on_batch(processed_input, label_categorical, return_dict=True)

            end_time = time.time()
            logging.info(f"Learn request processed in {end_time - start_time:.4f}s. Label: {label}, Loss: {metrics.get('loss', 'N/A'):.4f}, Acc: {metrics.get('accuracy', 'N/A'):.4f}")

            return jsonify({
                "message": "Model updated with the provided example.",
                "label_provided": label,
                "loss_on_example": metrics.get('loss'),
                "accuracy_on_example": metrics.get('accuracy')
            })

        except (ValueError, TypeError) as ve:
            logging.warning(f"Bad learn request data: {ve}")
            return jsonify({"error": f"Invalid input data: {ve}"}), 400
        except Exception as e:
            logging.error(f"Learning step error: {e}", exc_info=True)
            return jsonify({"error": "Learning step failed."}), 500

    return flask_app

# This check prevents running the app when imported, e.g., by Gunicorn
# Use run_api.py to start the server directly.
# if __name__ == '__main__':
#     app = create_app()
#     app.run(host=config.API_HOST, port=config.API_PORT, debug=False)