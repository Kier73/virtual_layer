import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted # To check if pipeline steps are fitted

class SurrogateModel:
    def __init__(self, name: str):
        self.name = name
        self.cost_model: Optional[Pipeline] = None
        self.fidelity_model: Optional[Pipeline] = None
        self.trained = False
        self._num_features_trained = 0 # To store the number of features used in training

    def train(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a NumPy array.")
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D array.")
        if data.shape[0] == 0: # Check for empty data (no samples)
            raise ValueError("Input data cannot be empty (no samples).")
        if data.shape[1] < 3:
            raise ValueError(f"Input data must have at least 3 columns (features, cost, fidelity), got {data.shape[1]}.")

        num_features = data.shape[1] - 2
        self._num_features_trained = num_features # Store for prediction checks

        X = data[:, :num_features]
        y_cost = data[:, num_features]
        y_fidelity = data[:, num_features + 1]

        # Ensure X is 2D. If num_features is 1, X will be (n_samples, 1) already due to slicing.
        # If num_features is 0 (data.shape[1] == 2, which is caught by data.shape[1] < 3),
        # this would mean X = data[:, :0] which is valid but perhaps not what's intended for typical models.
        # However, the check data.shape[1] < 3 ensures num_features >= 1.

        self.cost_model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])

        self.fidelity_model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge()) # Using Ridge for fidelity as specified
        ])

        self.cost_model.fit(X, y_cost)
        self.fidelity_model.fit(X, y_fidelity)
        self.trained = True

    def _check_trained_and_features(self, latent: np.ndarray):
        if not self.trained or self.cost_model is None or self.fidelity_model is None:
            # Check self.cost_model and self.fidelity_model for mypy, though self.trained should suffice
            raise RuntimeError("Surrogate model not trained")

        # Check if the pipeline steps are fitted (more robust than just self.trained flag)
        check_is_fitted(self.cost_model.named_steps['scaler'])
        check_is_fitted(self.cost_model.named_steps['regressor'])
        check_is_fitted(self.fidelity_model.named_steps['scaler'])
        check_is_fitted(self.fidelity_model.named_steps['regressor'])

        if not isinstance(latent, np.ndarray):
            raise TypeError("Input latent features must be a NumPy array.")

        # Use the stored _num_features_trained for consistent checking
        expected_features = self._num_features_trained

        # Handle latent being a 1D array for a single sample with multiple features,
        # or a 1D array for multiple samples if n_features_in_ is 1.
        if latent.ndim == 1:
            if expected_features == 1 and latent.size > 1: # multiple samples, 1 feature each
                if latent.size != expected_features : # This check is for single sample (1,N) vs (N)
                     pass # Will be reshaped to (latent.size, 1)
            elif latent.size != expected_features: # Single sample (N) vs expected N features
                 raise ValueError(f"Input latent vector has {latent.size} features, but model expects {expected_features}.")
        elif latent.ndim == 2: # Batch of samples
            if latent.shape[1] != expected_features:
                 raise ValueError(f"Input latent vector has {latent.shape[1]} features per sample, but model expects {expected_features}.")
        else: # Wrong number of dimensions
            raise ValueError(f"Input latent vector must be 1D or 2D, got {latent.ndim}D.")


    def predict_cost(self, latent: np.ndarray) -> float: # Prompt specifies -> float (single prediction)
        self._check_trained_and_features(latent)

        # Reshape latent to 2D (1, n_features) if it's a single 1D sample
        # or (n_samples, 1) if it's multiple 1D samples for a single-feature model.
        if latent.ndim == 1:
            if self._num_features_trained == 1:
                latent_reshaped = latent.reshape(-1, 1)
            else: # Single sample with multiple features
                latent_reshaped = latent.reshape(1, -1)
        else: # Already 2D
            latent_reshaped = latent

        try:
            prediction = self.cost_model.predict(latent_reshaped) # type: ignore
        except Exception as e:
            raise RuntimeError(f"Error during cost prediction: {e}")

        # If multiple samples were passed in latent_reshaped (e.g. for a 1-feature model)
        # the prompt implies predict_cost returns a single float.
        # This might mean it's intended for single sample predictions at a time,
        # or the return type hint in the prompt is for the common case of one sample.
        # For now, returning the first prediction if multiple were made.
        return float(prediction[0])


    def predict_fidelity(self, latent: np.ndarray) -> float: # Prompt specifies -> float
        if not self.trained or self.cost_model is None or self.fidelity_model is None:
            # Fidelity model might not be trained if it's optional or trained separately
            # Prompt says "if not self.trained: return 1.0" - this implies a default
            # if the whole surrogate isn't trained.
            return 1.0 # Default fidelity if not trained

        self._check_trained_and_features(latent) # Will raise if not trained

        if latent.ndim == 1:
            if self._num_features_trained == 1:
                latent_reshaped = latent.reshape(-1, 1)
            else:
                latent_reshaped = latent.reshape(1, -1)
        else:
            latent_reshaped = latent

        try:
            prediction = self.fidelity_model.predict(latent_reshaped) # type: ignore
        except Exception as e:
            raise RuntimeError(f"Error during fidelity prediction: {e}")
        return float(prediction[0]) # Return first prediction if batch
```
