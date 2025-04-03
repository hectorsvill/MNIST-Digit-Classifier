# tests/test_api.py
"""
Tests for the Flask API endpoints.
Uses the test_client fixture from conftest.py.
"""
import pytest
import json
from src import config # To get NUM_CLASSES etc.

def test_health_endpoint(test_client):
    """Test the /health endpoint."""
    response = test_client.get('/health')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['status'] == 'ok'
    assert 'Model is loaded' in json_data['message']

def test_predict_endpoint_success(test_client, valid_predict_input, mock_model):
    """Test /predict with valid input."""
    response = test_client.post('/predict', json=valid_predict_input)
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'prediction' in json_data
    assert 'confidence' in json_data
    # Check against the mock model's expected output (predicted class 3)
    assert json_data['prediction'] == 3
    assert json_data['confidence'] == pytest.approx(0.9) # Use approx for floats
    # Verify mock was called
    mock_model.predict.assert_called_once()


def test_predict_endpoint_missing_features(test_client):
    """Test /predict with missing 'features' key."""
    response = test_client.post('/predict', json={"wrong_key": []})
    assert response.status_code == 400
    json_data = response.get_json()
    assert 'error' in json_data
    assert "Missing 'features' key" in json_data['error']

def test_predict_endpoint_wrong_feature_count(test_client):
    """Test /predict with incorrect number of features."""
    response = test_client.post('/predict', json={"features": [0.0] * 100}) # Too few
    assert response.status_code == 400
    json_data = response.get_json()
    assert 'error' in json_data
    assert 'Invalid input data' in json_data['error']
    assert f"Expected {config.INPUT_SHAPE[0]} features" in json_data['error']

def test_predict_endpoint_not_json(test_client):
    """Test /predict with non-JSON data."""
    response = test_client.post('/predict', data="not json")
    assert response.status_code == 400
    json_data = response.get_json()
    assert 'error' in json_data
    assert "Request must be JSON" in json_data['error']

# --- Tests for /learn endpoint ---

def test_learn_endpoint_success(test_client, valid_learn_input, mock_model):
    """Test /learn with valid input."""
    response = test_client.post('/learn', json=valid_learn_input)
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'message' in json_data
    assert 'Model updated' in json_data['message']
    assert json_data['label_provided'] == valid_learn_input['label']
    assert 'loss_on_example' in json_data
    assert 'accuracy_on_example' in json_data
    # Check against mock train_on_batch output
    assert json_data['loss_on_example'] == pytest.approx(0.123)
    assert json_data['accuracy_on_example'] == pytest.approx(1.0)
    # Verify mock was called
    mock_model.train_on_batch.assert_called_once()


def test_learn_endpoint_missing_label(test_client, valid_predict_input): # Reuse predict input
    """Test /learn with missing 'label' key."""
    response = test_client.post('/learn', json=valid_predict_input) # Missing 'label'
    assert response.status_code == 400
    json_data = response.get_json()
    assert 'error' in json_data
    assert "Missing 'features' or 'label' key" in json_data['error']

def test_learn_endpoint_invalid_label(test_client, valid_learn_input):
    """Test /learn with an out-of-range label."""
    bad_input = valid_learn_input.copy()
    bad_input['label'] = config.NUM_CLASSES + 5 # Invalid label
    response = test_client.post('/learn', json=bad_input)
    assert response.status_code == 400
    json_data = response.get_json()
    assert 'error' in json_data
    assert 'Invalid input data' in json_data['error']
    assert 'Label must be an integer between' in json_data['error']

def test_learn_endpoint_wrong_features(test_client, valid_learn_input):
    """Test /learn with incorrect feature count."""
    bad_input = valid_learn_input.copy()
    bad_input['features'] = [0.1] * 50 # Wrong number
    response = test_client.post('/learn', json=bad_input)
    assert response.status_code == 400
    json_data = response.get_json()
    assert 'error' in json_data
    assert 'Invalid input data' in json_data['error']
    assert f"Expected {config.INPUT_SHAPE[0]} features" in json_data['error']