# tests/conftest.py
import pytest
import numpy as np
from unittest.mock import MagicMock

# Import the factory function from the source code
from src.api.app import create_app
from src import config

@pytest.fixture(scope='module') # Keep mock_model module-scoped if it doesn't depend on function-scoped fixtures
def mock_model():
    """Fixture to create a mock Keras model."""
    # ... (mock_model code remains the same) ...
    mock = MagicMock()
    def mock_predict(data, verbose=0):
        if data.shape == (1, config.INPUT_SHAPE[0]):
            probs = np.zeros((1, config.NUM_CLASSES), dtype=np.float32)
            probs[0, 3] = 0.9
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

# CHANGE SCOPE HERE: from 'module' to 'function' (or remove scope= argument)
@pytest.fixture # Default scope is 'function'
def test_client(mocker, mock_model): # mocker is function-scoped
    """Fixture to create a test client for the Flask app, mocking model loading."""

    # --- Mock load_trained_model ---
    mocker.patch('src.api.app.load_trained_model', return_value=True)

    # --- Mock the global model variable AFTER patching load_model ---
    mocker.patch('src.api.app.model', mock_model)
    mock_optimizer = MagicMock()
    mocker.patch('src.api.app.optimizer', mock_optimizer)

    # --- Create the app using the factory ---
    app = create_app()
    app.config['TESTING'] = True

    # --- Create and return the test client ---
    with app.test_client() as client:
        yield client

# Fixture for sample valid input data (can remain session-scoped)
@pytest.fixture(scope='session')
def valid_predict_input():
    return {"features": [0.0] * config.INPUT_SHAPE[0]}

@pytest.fixture(scope='session')
def valid_learn_input():
    return {"features": [1.0] * config.INPUT_SHAPE[0], "label": 5}