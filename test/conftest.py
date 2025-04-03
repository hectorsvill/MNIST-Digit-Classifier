# tests/conftest.py
import pytest
from unittest.mock import MagicMock
import sys
import os
import numpy as np # Add numpy import

# Ensure the app directory is in the Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src import config # Import config after adjusting path



@pytest.fixture(scope="function")
def mock_process_uploaded_image(mocker):
    """Fixture to mock the process_uploaded_image function used in app.py"""
    # Target the function where it's *used* within the app module
    mock = mocker.patch("src.api.app.process_uploaded_image")
    # Provide a default return value (can be overridden in tests if needed)
    default_features = np.zeros((config.INPUT_SHAPE[0],), dtype=np.float32)
    mock.return_value = default_features
    return mock # Return the mock object itself

@pytest.fixture(scope='module')
def mock_model():
    """Fixture to create a mock Keras model."""
    # ... (mock_model code remains the same) ...
    mock = MagicMock()
    def mock_predict(data, verbose=0):
        if data.shape == (1, config.INPUT_SHAPE[0]):
            probs = np.zeros((1, config.NUM_CLASSES), dtype=np.float32)
            probs[0, 3] = 0.9 # Predicts 3
            probs[0, 1] = 0.1
            return probs
        else:
            raise ValueError("Mock predict received unexpected shape")
    mock.predict.side_effect = mock_predict
    def mock_train_on_batch(features, labels, return_dict=True):
         return {'loss': 0.123, 'accuracy': 1.0}
    mock.train_on_batch.side_effect = mock_train_on_batch
    mock.compile = MagicMock()
    return mock

# CHANGE SCOPE HERE: from 'module' to 'function'
@pytest.fixture # Default scope is 'function'
def test_client(mocker, mock_model, mock_process_uploaded_image): # Add the new mock fixture here
    """Fixture to create a test client for the Flask app, mocking dependencies."""
    # Mock load_trained_model (as before)
    mocker.patch('src.api.app.load_trained_model', return_value=True)
    # Mock the global model variable (as before)
    mocker.patch('src.api.app.model', mock_model)
    mock_optimizer = MagicMock()
    mocker.patch('src.api.app.optimizer', mock_optimizer)
    # The mock_process_uploaded_image fixture already patches the function

    # Create the app using the factory
    from src.api.app import create_app # Import here to ensure patches are active
    app = create_app()
    app.config['TESTING'] = True

    # Create and return the test client
    with app.test_client() as client:
        yield client

# --- Other fixtures (valid_predict_input, etc.) ---
# These might not be needed anymore if tests use file uploads primarily
@pytest.fixture(scope='session')
def valid_predict_input(): # Keep if needed for other tests, otherwise remove
    return {"features": [0.0] * config.INPUT_SHAPE[0]}

@pytest.fixture(scope='session')
def valid_learn_input(): # Keep if needed for other tests, otherwise remove
    return {"features": [1.0] * config.INPUT_SHAPE[0], "label": 5}