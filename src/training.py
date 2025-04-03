# src/training.py
"""
Contains the logic for the custom training loop and evaluation.
"""
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from . import config # Use config from the same package
from . import data_loader # Use data_loader from the same package

# --- Metrics Setup ---
# We define these outside the function so they can be reused if needed,
# but they will be reset within the training function.
loss_fn = tf.keras.losses.CategoricalCrossentropy()
train_accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
val_accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

# --- Training Step Functions (using tf.function for speed) ---
@tf.function
def train_step(model, optimizer, images, labels):
    """Performs a single training step (forward pass, loss, gradients, update)."""
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # Update metrics
    train_loss_metric.update_state(loss)
    train_accuracy_metric.update_state(labels, predictions)

@tf.function
def validation_step(model, images, labels):
    """Performs a single validation step (forward pass, loss)."""
    predictions = model(images, training=False)
    loss = loss_fn(labels, predictions)
    # Update metrics
    val_loss_metric.update_state(loss)
    val_accuracy_metric.update_state(labels, predictions)

# --- Main Training Function ---
def run_training(model, train_dataset, val_dataset, epochs=config.EPOCHS, learning_rate=config.LEARNING_RATE):
    """Runs the custom training loop."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

    print(f"\nStarting training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        print(f"\n--- Starting Epoch {epoch + 1}/{epochs} ---")
        epoch_start_time = time.time()

        # Reset metrics at the start of each epoch
        train_loss_metric.reset_state()
        train_accuracy_metric.reset_state()
        val_loss_metric.reset_state()
        val_accuracy_metric.reset_state()

        # Training loop
        for step, (train_images, train_labels) in enumerate(train_dataset):
            train_step(model, optimizer, train_images, train_labels)
            # Optional: Print batch progress
            # if step % 100 == 0:
            #     print(f"  Batch {step}: Train Loss: {train_loss_metric.result():.4f}, Train Acc: {train_accuracy_metric.result():.4f}")

        epoch_train_loss = train_loss_metric.result()
        epoch_train_accuracy = train_accuracy_metric.result()
        history['train_loss'].append(epoch_train_loss.numpy())
        history['train_accuracy'].append(epoch_train_accuracy.numpy())

        # Validation loop
        for val_images, val_labels in val_dataset:
            validation_step(model, val_images, val_labels)

        epoch_val_loss = val_loss_metric.result()
        epoch_val_accuracy = val_accuracy_metric.result()
        history['val_loss'].append(epoch_val_loss.numpy())
        history['val_accuracy'].append(epoch_val_accuracy.numpy())

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} finished in {epoch_duration:.2f}s")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}")
        print(f"  Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_accuracy:.4f}")

    total_training_time = time.time() - start_time
    print(f"\nTraining finished in {total_training_time:.2f} seconds.")
    return history

# --- Plotting Function ---
def plot_history(history):
    """Plots training and validation loss and accuracy."""
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], 'ro-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs_range, history['val_accuracy'], 'ro-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy During Training')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# --- Evaluation Function ---
def evaluate_model(model, test_dataset, x_test_norm, y_test_original_labels):
    """Evaluates the model on the test set and prints metrics."""
    print("\n--- Evaluating on the Test Set ---")
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Detailed report and confusion matrix
    print("\nCalculating detailed metrics...")
    y_pred_probs = model.predict(x_test_norm)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    num_classes = len(np.unique(y_test_original_labels)) # Get number of classes from actual labels

    print("\nClassification Report:")
    print(classification_report(y_test_original_labels, y_pred_classes, target_names=[str(i) for i in range(num_classes)]))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_original_labels, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()