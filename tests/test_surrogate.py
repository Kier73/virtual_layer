import pytest
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge

from virtual_layer.surrogate import BasicSklearnSurrogate, SurrogateModel

# --- Fixture for synthetic data ---
@pytest.fixture
def synthetic_data():
    """
    Generates synthetic data for testing surrogate models.
    Returns a tuple: (features, costs, fidelities)
    """
    rng = np.random.RandomState(42) # Use a fixed seed for reproducibility
    n_samples, n_features = 100, 3
    features = rng.rand(n_samples, n_features)

    # Define some arbitrary linear relationships for costs and fidelities
    # Costs = 2*f0 + 0.5*f1 - f2 + noise
    costs = 2 * features[:, 0] + 0.5 * features[:, 1] - features[:, 2] + rng.randn(n_samples) * 0.1

    # Fidelities = 0.8 - 0.3*f0 + 0.1*f2 + noise, clipped between 0 and 1
    fidelities = 0.8 - 0.3 * features[:, 0] + 0.1 * features[:, 2] + rng.randn(n_samples) * 0.05
    fidelities = np.clip(fidelities, 0, 1)

    return features, costs, fidelities

# --- Tests for BasicSklearnSurrogate ---

@pytest.mark.parametrize("scale_targets", [True, False])
def test_initialization(scale_targets):
    model = BasicSklearnSurrogate(scale_targets=scale_targets)
    assert isinstance(model.cost_model, BayesianRidge)
    assert isinstance(model.fidelity_model, BayesianRidge)
    assert isinstance(model.feature_scaler, StandardScaler)
    assert model.is_trained is False

    if scale_targets:
        assert isinstance(model.cost_target_scaler, StandardScaler)
        assert isinstance(model.fidelity_target_scaler, StandardScaler)
    else:
        assert model.cost_target_scaler is None
        assert model.fidelity_target_scaler is None

@pytest.mark.parametrize("scale_targets", [True, False])
def test_train(synthetic_data, scale_targets):
    features, costs, fidelities = synthetic_data
    model = BasicSklearnSurrogate(scale_targets=scale_targets)

    model.train(features, costs, fidelities)

    assert model.is_trained is True
    check_is_fitted(model.feature_scaler)
    check_is_fitted(model.cost_model)
    check_is_fitted(model.fidelity_model)

    if scale_targets:
        check_is_fitted(model.cost_target_scaler)
        check_is_fitted(model.fidelity_target_scaler)

@pytest.mark.parametrize("scale_targets", [True, False])
def test_predict_before_training(scale_targets):
    model = BasicSklearnSurrogate(scale_targets=scale_targets)
    dummy_features = np.random.rand(10, 3)

    with pytest.raises(RuntimeError, match="Surrogate model has not been trained yet."):
        model.predict_cost(dummy_features)

    with pytest.raises(RuntimeError, match="Surrogate model has not been trained yet."):
        model.predict_fidelity(dummy_features)

@pytest.mark.parametrize("scale_targets", [True, False])
def test_predict_cost_after_training(synthetic_data, scale_targets):
    features, costs, _ = synthetic_data # Fidelities not used directly in this test
    model = BasicSklearnSurrogate(scale_targets=scale_targets)
    model.train(features, costs, np.random.rand(len(costs))) # Dummy fidelities for training call

    # Test with multiple samples
    predictions = model.predict_cost(features)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (features.shape[0],) # Should be a 1D array of predictions

    # Optional: Check for some correlation if relationship is simple enough
    # This is a very loose check, as model is not expected to be perfect.
    # correlation = np.corrcoef(costs, predictions)[0, 1]
    # assert correlation > 0.5 # Example threshold, highly dependent on data & model

    # Test with a single sample
    single_feature_sample = features[0, :].reshape(1, -1) # Ensure 2D for scaler
    single_prediction = model.predict_cost(single_feature_sample)
    assert isinstance(single_prediction, np.ndarray)
    assert single_prediction.shape == (1,)

    # Test with a single sample passed as 1D array (model should handle reshape)
    single_feature_sample_1d = features[0, :]
    single_prediction_1d = model.predict_cost(single_feature_sample_1d)
    assert isinstance(single_prediction_1d, np.ndarray)
    assert single_prediction_1d.shape == (1,)
    assert np.allclose(single_prediction, single_prediction_1d)


@pytest.mark.parametrize("scale_targets", [True, False])
def test_predict_fidelity_after_training(synthetic_data, scale_targets):
    features, _, fidelities = synthetic_data # Costs not used directly
    model = BasicSklearnSurrogate(scale_targets=scale_targets)
    model.train(features, np.random.rand(len(fidelities)), fidelities) # Dummy costs

    # Test with multiple samples
    predictions = model.predict_fidelity(features)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (features.shape[0],)

    # Test with a single sample (2D)
    single_feature_sample = features[0, :].reshape(1, -1)
    single_prediction = model.predict_fidelity(single_feature_sample)
    assert isinstance(single_prediction, np.ndarray)
    assert single_prediction.shape == (1,)

    # Test with a single sample (1D)
    single_feature_sample_1d = features[0, :]
    single_prediction_1d = model.predict_fidelity(single_feature_sample_1d)
    assert isinstance(single_prediction_1d, np.ndarray)
    assert single_prediction_1d.shape == (1,)
    assert np.allclose(single_prediction, single_prediction_1d)

def test_predict_with_1d_feature_in_train_and_predict(scale_targets=False):
    # Test case where features might be 1D (e.g. single feature)
    rng = np.random.RandomState(42)
    n_samples = 50
    features_1d = rng.rand(n_samples) # 1D features
    costs_1d = 2 * features_1d + rng.randn(n_samples) * 0.1
    fidelities_1d = 0.5 - 0.2 * features_1d + rng.randn(n_samples) * 0.05

    model = BasicSklearnSurrogate(scale_targets=scale_targets)
    model.train(features_1d, costs_1d, fidelities_1d) # Train with 1D features

    # Predict with 1D features (multiple samples)
    predictions_cost = model.predict_cost(features_1d)
    assert predictions_cost.shape == (n_samples,)

    # Predict with a single 1D feature sample (passed as scalar-like)
    # Note: The model's predict methods internally reshape a 1D input features array to 2D (1, n_features)
    # if it's a single sample. If features_1d[0] (a scalar) is passed, it needs to be wrapped.
    single_pred_cost = model.predict_cost(np.array([features_1d[0]]))
    assert single_pred_cost.shape == (1,)

    single_pred_cost_reshaped = model.predict_cost(features_1d[0].reshape(1, -1)) # explicit 2D
    assert single_pred_cost_reshaped.shape == (1,)
    assert np.allclose(single_pred_cost, single_pred_cost_reshaped)


# Ensure SurrogateModel is an ABC and cannot be instantiated directly
def test_surrogate_model_is_abc():
    with pytest.raises(TypeError, match="Can't instantiate abstract class SurrogateModel"):
        SurrogateModel()

# Test that train method handles 1D targets correctly
@pytest.mark.parametrize("scale_targets", [True, False])
def test_train_with_1d_targets(synthetic_data, scale_targets):
    features, costs, fidelities = synthetic_data
    model = BasicSklearnSurrogate(scale_targets=scale_targets)

    # Ensure costs and fidelities are 1D for this test
    costs_1d = costs.ravel()
    fidelities_1d = fidelities.ravel()

    model.train(features, costs_1d, fidelities_1d) # Should internally reshape targets

    assert model.is_trained
    # Make a prediction to ensure model is usable
    predictions = model.predict_cost(features)
    assert predictions.shape == (features.shape[0],)
