# tests/test_api.py
import json
import pytest
import io
from PIL import Image
import numpy as np
from unittest.mock import ANY

from src import config
# No longer need to import preprocessing directly for mocking here
# from src.api import preprocessing

# Helper function (keep as is)
def create_dummy_png_bytes(width=28, height=28, color=0):
    img = Image.new('L', (width, height), color=color)
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG')
    byte_arr.seek(0)
    return byte_arr.getvalue()

# --- Test Health Endpoint (remains the same) ---
def test_health_endpoint(test_client):
    response = test_client.get('/health')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['status'] == 'ok'

# --- Test Predict Endpoint ---

# Add the mock_process_uploaded_image fixture to the test signature
def test_predict_endpoint_success(test_client, mock_model, mock_process_uploaded_image):
    """Test /predict with a valid image upload."""
    # Arrange
    dummy_image_bytes = create_dummy_png_bytes(color=128)
    # Configure the mock return value for this specific test
    mock_processed_features = np.full((config.INPUT_SHAPE[0],), 128.0/255.0, dtype=np.float32)
    mock_process_uploaded_image.return_value = mock_processed_features

    # Configure mock model prediction for this test
    mock_model.predict.return_value = np.array([[0.0, 0.1, 0.1, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32) # Predict 3

    # Act
    data = {'image': (io.BytesIO(dummy_image_bytes), 'test_digit.png')}
    response = test_client.post('/predict', data=data, content_type='multipart/form-data')

    # Assert
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['prediction'] == 3
    assert json_data['confidence'] == pytest.approx(0.9)

    # Verify mocks using the fixture variable
    mock_process_uploaded_image.assert_called_once() # Assert on the fixture
    mock_model.predict.assert_called_once()
    call_args, _ = mock_model.predict.call_args
    assert call_args[0].shape == (1, config.INPUT_SHAPE[0])
    np.testing.assert_allclose(call_args[0][0], mock_processed_features)


def test_predict_endpoint_invalid_image(test_client, mock_process_uploaded_image): # Add fixture
    """Test /predict with a file that is not a valid image."""
    # Arrange: Configure the mock fixture to raise an error
    error_message = "Invalid image format"
    mock_process_uploaded_image.side_effect = ValueError(error_message)

    # Act
    data = {'image': (io.BytesIO(b'this is not an image'), 'test.txt')}
    response = test_client.post('/predict', data=data, content_type='multipart/form-data')

    # Assert
    assert response.status_code == 400
    json_data = response.get_json()
    assert error_message in json_data['error']
    mock_process_uploaded_image.assert_called_once() # Assert on the fixture


# --- Tests for /learn endpoint ---

# Add the mock_process_uploaded_image fixture
def test_learn_endpoint_success(test_client, mock_model, mock_process_uploaded_image):
    """Test /learn with valid image and label."""
    # Arrange
    dummy_image_bytes = create_dummy_png_bytes(color=64)
    correct_label = 3
    # Configure mock return value
    mock_processed_features = np.full((config.INPUT_SHAPE[0],), 64.0/255.0, dtype=np.float32)
    mock_process_uploaded_image.return_value = mock_processed_features
    # Configure mock train_on_batch
    mock_model.train_on_batch.return_value = {'loss': 0.123, 'accuracy': 1.0}

    # Act
    data = {
        'image': (io.BytesIO(dummy_image_bytes), 'learn_digit.png'),
        'label': str(correct_label)
    }
    response = test_client.post('/learn', data=data, content_type='multipart/form-data')

    # Assert
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['label_provided'] == correct_label
    assert json_data['loss_on_example'] == pytest.approx(0.123)
    assert json_data['accuracy_on_example'] == pytest.approx(1.0)

    # Verify mocks
    mock_process_uploaded_image.assert_called_once() # Assert on the fixture
    mock_model.train_on_batch.assert_called_once()
    # ... rest of verification ...


# Add the mock_process_uploaded_image fixture
def test_learn_endpoint_image_processing_error(test_client, mock_process_uploaded_image):
    """Test /learn when image processing fails."""
    # Arrange: Configure mock fixture to raise error
    error_message = "Bad image data"
    mock_process_uploaded_image.side_effect = ValueError(error_message)

    # Act
    data = {
        'image': (io.BytesIO(b'bad data'), 'bad.png'),
        'label': '1'
    }
    response = test_client.post('/learn', data=data, content_type='multipart/form-data')

    # Assert
    assert response.status_code == 400
    json_data = response.get_json()
    assert error_message in json_data['error']
    mock_process_uploaded_image.assert_called_once() # Assert on the fixture

# --- Other tests (missing file/label, invalid label) remain largely the same ---
# Ensure they don't rely on the mock_process_uploaded_image fixture if not needed,
# or accept it if their setup requires the test_client which now depends on it.

def test_learn_endpoint_missing_label_field(test_client): # Doesn't need the mock fixture directly
    dummy_image_bytes = create_dummy_png_bytes()
    data = {'image': (io.BytesIO(dummy_image_bytes), 'learn_digit.png')}
    response = test_client.post('/learn', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert "Missing 'label' form field" in response.get_json()['error']

def test_learn_endpoint_missing_image_file(test_client): # Doesn't need the mock fixture directly
    data = {'label': '5'}
    response = test_client.post('/learn', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert "No image file part" in response.get_json()['error']

def test_learn_endpoint_invalid_label_value(test_client): # Doesn't need the mock fixture directly
    dummy_image_bytes = create_dummy_png_bytes()
    data_non_digit = {'image': (io.BytesIO(dummy_image_bytes), 'learn_digit.png'), 'label': 'abc'}
    response_non_digit = test_client.post('/learn', data=data_non_digit, content_type='multipart/form-data')
    assert response_non_digit.status_code == 400
    assert "Label must be an integer digit" in response_non_digit.get_json()['error']

    data_out_range = {'image': (io.BytesIO(dummy_image_bytes), 'learn_digit.png'), 'label': '15'}
    response_out_range = test_client.post('/learn', data=data_out_range, content_type='multipart/form-data')
    assert response_out_range.status_code == 400
    assert "Label must be an integer between" in response_out_range.get_json()['error']