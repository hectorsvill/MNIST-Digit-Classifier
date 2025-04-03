# train.py
"""
Main script to execute the training process.
"""
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import os # Import os for savefig path

# Import from our source directory
from src import config
from src import data_loader
from src import model as model_builder # Avoid name clash with tf.keras.models
from src import training

# --- Matplotlib Backend Configuration (Fix for UserWarning) ---
# Try to use a non-interactive backend if just saving plots
# Put this near the top imports
try:
    # Check if running in a non-GUI environment (like a basic terminal)
    # This is a simple check; more robust checks might be needed
    if os.environ.get('DISPLAY', '') == '':
        print("No display found. Using non-interactive Agg backend for Matplotlib.")
        plt.switch_backend('Agg')
    else:
        print("Display found. Using default Matplotlib backend.")
except ImportError:
    print("Could not switch Matplotlib backend.")
# -------------------------------------------------------------


def main(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE, save_path=config.MODEL_SAVE_PATH):
    """Loads data, builds model, trains, evaluates, and saves."""

    # 1. Load and Prepare Data
    (x_train, y_train), (x_val, y_val), (x_test, y_test_cat), (x_test_orig, y_test_orig) = \
        data_loader.load_and_preprocess_mnist()

    # Create tf.data datasets
    train_dataset = data_loader.create_tf_dataset(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_dataset = data_loader.create_tf_dataset(x_val, y_val, batch_size=batch_size)
    test_dataset = data_loader.create_tf_dataset(x_test, y_test_cat, batch_size=batch_size) # Use categorical for model.evaluate

    # 2. Build Model
    model = model_builder.build_our_cool_model()
    model.summary() # Print model structure

    # --- COMPILE THE MODEL HERE ---
    # Even though we use a custom training loop, model.evaluate needs compilation.
    print("Compiling model for evaluation...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr), # Optimizer needed for compile, though not used by evaluate
        loss=tf.keras.losses.CategoricalCrossentropy(),       # Must match the loss used conceptually
        metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')] # Metrics to report during evaluate
    )
    print("Model compiled.")
    # -----------------------------

    # 3. Train Model (using custom loop)
    history = training.run_training(model, train_dataset, val_dataset, epochs=epochs, learning_rate=lr)

    # 4. Plot Training History (and save instead of showing)
    print("Plotting and saving training history...")
    training.plot_history(history) # Keep the function name
    # The saving logic will be inside plot_history now (see below)

    # 5. Evaluate Model
    # Pass x_test (normalized) and y_test_orig (integer labels) for detailed report
    training.evaluate_model(model, test_dataset, x_test, y_test_orig)
    # The saving logic for confusion matrix will be inside evaluate_model (see below)

    # 6. Save Model
    print(f"\nSaving trained model to: {save_path}")
    try:
        model.save(save_path)
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")

    # Remove plt.show() from here if plots are saved within functions
    # plt.show() # Remove or comment out


if __name__ == "__main__":
    # ... (rest of the argument parsing and GPU setup remains the same) ...
    parser = argparse.ArgumentParser(description="Train the MNIST digit classifier.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--save_path", type=str, default=config.MODEL_SAVE_PATH, help="Path to save the trained model.")

    args = parser.parse_args()

    # Set GPU memory growth if available (optional, good practice)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s), memory growth enabled.")
        except RuntimeError as e:
            print(f"Could not set memory growth for GPU: {e}")
    else:
        print("No GPU found, using CPU.")

    main(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_path=args.save_path)