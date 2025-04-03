# tests/test_data_loader.py
"""
Tests for data loading and preprocessing functions.
"""
import tensorflow as tf
import numpy as np
import pytest
from src import data_loader
from src import config
from src.api.preprocessing import preprocess_api_input # Test API preprocessing too

def test_load_and_preprocess_mnist():
    """Test the main data loading and preprocessing function."""
    (x_train, y_train), (x_val, y_val), (x_test, y_test_cat), (x_test_orig, y_test_orig) = \
        data_loader.load_and_preprocess_mnist()

    # Check shapes
    assert x_train.shape[1] == config.INPUT_SHAPE[0]
    assert x_val.shape[1] == config.INPUT_SHAPE[0]
    assert x_test.shape[1] == config.INPUT_SHAPE[0]
    assert y_train.shape[1] == config.NUM_CLASSES
    assert y_val.shape[1] == config.NUM_CLASSES
    assert y_test_cat.shape[1] == config.NUM_CLASSES
    assert len(y_test_orig.shape) == 1 # Original labels should be 1D

    # Check normalization (values between 0 and 1)
    assert np.min(x_train) >= 0.0 and np.max(x_train) <= 1.0
    assert np.min(x_val) >= 0.0 and np.max(x_val) <= 1.0
    assert np.min(x_test) >= 0.0 and np.max(x_test) <= 1.0

    # Check one-hot encoding (sum of each row should be 1)
    assert np.allclose(np.sum(y_train, axis=1), 1.0)
    assert np.allclose(np.sum(y_val, axis=1), 1.0)
    assert np.allclose(np.sum(y_test_cat, axis=1), 1.0)

    # Check split sizes (approximate, depends on exact MNIST size)
    assert x_train.shape[0] + x_val.shape[0] == 60000
    assert x_val.shape[0] == config.VALIDATION_SPLIT
    assert x_test.shape[0] == 10000

def test_create_tf_dataset():
    """Test the tf.data.Dataset creation utility."""
    dummy_features = np.random.rand(100, config.INPUT_SHAPE[0]).astype(np.float32)
    dummy_labels = np.eye(config.NUM_CLASSES)[np.random.randint(0, config.NUM_CLASSES, 100)].astype(np.float32)

    dataset = data_loader.create_tf_dataset(dummy_features, dummy_labels, batch_size=10)

    # Check element spec (structure and types)
    feature_spec, label_spec = dataset.element_spec
    assert feature_spec.shape.as_list() == [None, config.INPUT_SHAPE[0]] # Batch dim is None
    assert label_spec.shape.as_list() == [None, config.NUM_CLASSES]
    assert feature_spec.dtype == tf.float32
    assert label_spec.dtype == tf.float32

    # Check iteration and batch size
    for batch_features, batch_labels in dataset.take(1): # Take one batch
        assert batch_features.shape[0] == 10
        assert batch_labels.shape[0] == 10
        break # Only need one batch

# --- Tests for API Preprocessing ---

def test_preprocess_api_input_valid_normalized():
    """Test API preprocessing with already normalized data."""
    valid_input = [0.5] * config.INPUT_SHAPE[0]
    processed = preprocess_api_input(valid_input)
    assert processed.shape == (1, config.INPUT_SHAPE[0])
    assert np.allclose(processed, 0.5)

def test_preprocess_api_input_valid_unnormalized():
    """Test API preprocessing with 0-255 data."""
    valid_input = [127.5] * config.INPUT_SHAPE[0] # Should be normalized to 0.5
    processed = preprocess_api_input(valid_input)
    assert processed.shape == (1, config.INPUT_SHAPE[0])
    assert np.allclose(processed, 0.5)


# ... other test functions ...

def test_preprocess_api_input_clamping():
    """Test API preprocessing clamps out-of-range values."""
    # Test negative values
    invalid_input_neg = [-10.0] * config.INPUT_SHAPE[0]
    # --- Make sure this line exists and uses 'processed' ---
    processed = preprocess_api_input(invalid_input_neg)
    assert processed.shape == (1, config.INPUT_SHAPE[0])
    # --- Make sure this line uses 'processed' ---
    assert np.allclose(processed, 0.0), f"Expected clamping to 0.0, but got {processed.min()}"

    # Test values > 1.0 (after the fix in preprocessing.py)
    invalid_input_high = [2.0] * config.INPUT_SHAPE[0]
    # --- Make sure this line exists and uses 'processed_high' ---
    processed_high = preprocess_api_input(invalid_input_high)
    assert processed_high.shape == (1, config.INPUT_SHAPE[0])
    # --- Make sure this line uses 'processed_high' ---
    assert np.allclose(processed_high, 1.0), f"Expected clamping to 1.0, but got {processed_high.max()}"

    # Test values > 255 (should normalize then clamp to 1.0)
    invalid_input_over_255 = [300.0] * config.INPUT_SHAPE[0]
    # --- Make sure this line exists and uses 'processed_over_255' ---
    processed_over_255 = preprocess_api_input(invalid_input_over_255)
    assert processed_over_255.shape == (1, config.INPUT_SHAPE[0])
    # --- Make sure this line uses 'processed_over_255' ---
    assert np.allclose(processed_over_255, 1.0), f"Expected >255 input to clamp to 1.0 after norm, but got {processed_over_255.max()}"

# ... rest of the file ...    # ... test for negative values (likely okay) ...
    assert np.allclose(processed, 0.0)

    invalid_input_high = [2.0] * config.INPUT_SHAPE[0] # Input > 1.0
    processed_high = preprocess_api_input(invalid_input_high)
    assert processed_high.shape == (1, config.INPUT_SHAPE[0])
    # --- THIS ASSERTION IS LIKELY FAILING ---
    assert np.allclose(processed_high, 1.0) # Expects clamping to 1.0

def test_preprocess_api_input_wrong_count():
    """Test API preprocessing with wrong number of features."""
    invalid_input = [0.5] * 100
    with pytest.raises(ValueError, match=f"Expected {config.INPUT_SHAPE[0]} features"):
        preprocess_api_input(invalid_input)

def test_preprocess_api_input_wrong_type():
    """Test API preprocessing with wrong input type."""
    invalid_input = "not a list"
    with pytest.raises(TypeError, match="must be a list or NumPy array"):
        preprocess_api_input(invalid_input)