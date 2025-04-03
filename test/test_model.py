# tests/test_model.py
"""
Tests for the model building function.
"""
import tensorflow as tf
from src import model as model_builder
from src import config

def test_build_model_structure():
    """Test if the model builds and has correct input/output shapes."""
    model = model_builder.build_our_cool_model()

    assert isinstance(model, tf.keras.models.Sequential)
    assert len(model.layers) > 2 # Should have input, hidden, output

    # Check input shape (accessing through _feed_input_shapes or build)
    # Model needs to be built first to know input shape reliably
    # We can build it by passing dummy data or calling build()
    model.build(input_shape=(None, config.INPUT_SHAPE[0])) # Batch dim is None
    assert model.input_shape == (None, config.INPUT_SHAPE[0])

    # Check output shape
    assert model.output_shape == (None, config.NUM_CLASSES)

    # Check output activation
    assert model.layers[-1].activation == tf.keras.activations.softmax