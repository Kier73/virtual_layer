import abc
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted # For checking if model/scaler is fitted

class SurrogateModel(abc.ABC):
    @abc.abstractmethod
    def train(self, features: np.ndarray, targets_cost: np.ndarray, targets_fidelity: np.ndarray):
        """
        Trains the surrogate model(s).

        Args:
            features (np.ndarray): Input features for training.
            targets_cost (np.ndarray): Cost target values.
            targets_fidelity (np.ndarray): Fidelity target values.
        """
        pass

    @abc.abstractmethod
    def predict_cost(self, features: np.ndarray) -> np.ndarray:
        """
        Predicts cost values for given features.

        Args:
            features (np.ndarray): Input features for prediction.

        Returns:
            np.ndarray: Predicted cost values.
        """
        pass

    @abc.abstractmethod
    def predict_fidelity(self, features: np.ndarray) -> np.ndarray:
        """
        Predicts fidelity values for given features.

        Args:
            features (np.ndarray): Input features for prediction.

        Returns:
            np.ndarray: Predicted fidelity values.
        """
        pass


class BasicSklearnSurrogate(SurrogateModel):
    def __init__(self, scale_targets: bool = False): # Added option to control target scaling
        self.cost_model = BayesianRidge()
        self.fidelity_model = BayesianRidge()

        self.feature_scaler = StandardScaler()

        self.scale_targets = scale_targets
        if self.scale_targets:
            self.cost_target_scaler = StandardScaler()
            self.fidelity_target_scaler = StandardScaler()
        else:
            self.cost_target_scaler = None
            self.fidelity_target_scaler = None

        self.is_trained = False

    def train(self, features: np.ndarray, targets_cost: np.ndarray, targets_fidelity: np.ndarray):
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        if targets_cost.ndim == 1:
            targets_cost_reshaped = targets_cost.reshape(-1, 1)
        else:
            targets_cost_reshaped = targets_cost

        if targets_fidelity.ndim == 1:
            targets_fidelity_reshaped = targets_fidelity.reshape(-1, 1)
        else:
            targets_fidelity_reshaped = targets_fidelity

        # Fit and transform features
        scaled_features = self.feature_scaler.fit_transform(features)

        # Prepare targets
        processed_targets_cost = targets_cost_reshaped
        if self.scale_targets and self.cost_target_scaler:
            processed_targets_cost = self.cost_target_scaler.fit_transform(targets_cost_reshaped)

        processed_targets_fidelity = targets_fidelity_reshaped
        if self.scale_targets and self.fidelity_target_scaler:
            processed_targets_fidelity = self.fidelity_target_scaler.fit_transform(targets_fidelity_reshaped)

        # Train models (scikit-learn expects 1D targets for these models)
        self.cost_model.fit(scaled_features, processed_targets_cost.ravel())
        self.fidelity_model.fit(scaled_features, processed_targets_fidelity.ravel())

        self.is_trained = True

    def _check_is_trained(self):
        if not self.is_trained:
            raise RuntimeError("Surrogate model has not been trained yet. Call train() first.")
        # Optionally check if scalers and models are fitted using check_is_fitted
        # from sklearn.utils.validation, though self.is_trained flag mostly covers this.
        check_is_fitted(self.feature_scaler)
        check_is_fitted(self.cost_model)
        check_is_fitted(self.fidelity_model)
        if self.scale_targets:
            if self.cost_target_scaler: check_is_fitted(self.cost_target_scaler)
            if self.fidelity_target_scaler: check_is_fitted(self.fidelity_target_scaler)


    def predict_cost(self, features: np.ndarray) -> np.ndarray:
        self._check_is_trained()

        # Reshape features if necessary to be 2D
        if features.ndim == 1:
            if self.feature_scaler.n_features_in_ == 1:
                # Input is 1D array, scaler expects 1 feature: treat as (n_samples, 1)
                features = features.reshape(-1, 1)
            else:
                # Input is 1D array, scaler expects N features: treat as (1, N) (single sample)
                features = features.reshape(1, -1)

        scaled_features = self.feature_scaler.transform(features)
        predictions = self.cost_model.predict(scaled_features) # Predict returns 1D array

        if self.scale_targets and self.cost_target_scaler:
            # Inverse transform expects 2D array, predict gives 1D
            predictions_reshaped = predictions.reshape(-1, 1)
            predictions = self.cost_target_scaler.inverse_transform(predictions_reshaped).ravel()

        return predictions

    def predict_fidelity(self, features: np.ndarray) -> np.ndarray:
        self._check_is_trained()

        # Reshape features if necessary to be 2D
        if features.ndim == 1:
            if self.feature_scaler.n_features_in_ == 1:
                # Input is 1D array, scaler expects 1 feature: treat as (n_samples, 1)
                features = features.reshape(-1, 1)
            else:
                # Input is 1D array, scaler expects N features: treat as (1, N) (single sample)
                features = features.reshape(1, -1)

        scaled_features = self.feature_scaler.transform(features)
        predictions = self.fidelity_model.predict(scaled_features)

        if self.scale_targets and self.fidelity_target_scaler:
            predictions_reshaped = predictions.reshape(-1, 1)
            predictions = self.fidelity_target_scaler.inverse_transform(predictions_reshaped).ravel()

        return predictions

if __name__ == '__main__':
    # Example Usage
    # Generate some synthetic data
    rng = np.random.RandomState(0)
    n_samples, n_features = 100, 3
    X = rng.rand(n_samples, n_features)

    # Simple linear relationships for cost and fidelity
    true_cost_coeffs = np.array([2.0, 0.5, -1.0])
    true_fidelity_coeffs = np.array([-0.3, 0.1, 0.5])

    y_cost = np.dot(X, true_cost_coeffs) + rng.randn(n_samples) * 0.1
    y_fidelity = 0.8 + np.dot(X, true_fidelity_coeffs) + rng.randn(n_samples) * 0.05
    # Ensure fidelity is generally within a certain range, e.g., [0, 1] if it represents that
    y_fidelity = np.clip(y_fidelity, 0, 1)

    # Instantiate and train the surrogate model
    surrogate_no_target_scaling = BasicSklearnSurrogate(scale_targets=False)
    print("Training surrogate without target scaling...")
    surrogate_no_target_scaling.train(X, y_cost, y_fidelity)
    print("Training complete.")

    surrogate_with_target_scaling = BasicSklearnSurrogate(scale_targets=True)
    print("\nTraining surrogate with target scaling...")
    surrogate_with_target_scaling.train(X, y_cost, y_fidelity)
    print("Training complete.")

    # Make predictions
    X_test = rng.rand(10, n_features)

    print("\n--- Predictions (no target scaling) ---")
    cost_preds_no_scale = surrogate_no_target_scaling.predict_cost(X_test)
    fidelity_preds_no_scale = surrogate_no_target_scaling.predict_fidelity(X_test)
    for i in range(5): # Print first 5 test predictions
        print(f"Sample {i}: Predicted Cost={cost_preds_no_scale[i]:.2f}, Predicted Fidelity={fidelity_preds_no_scale[i]:.2f}")

    print("\n--- Predictions (with target scaling) ---")
    cost_preds_scale = surrogate_with_target_scaling.predict_cost(X_test)
    fidelity_preds_scale = surrogate_with_target_scaling.predict_fidelity(X_test)
    for i in range(5):
        print(f"Sample {i}: Predicted Cost={cost_preds_scale[i]:.2f}, Predicted Fidelity={fidelity_preds_scale[i]:.2f}")

    # Test single sample prediction
    single_sample = X[0, :]
    print(f"\n--- Single sample prediction (features: {single_sample}) ---")
    cost_single_no_scale = surrogate_no_target_scaling.predict_cost(single_sample)
    fidelity_single_no_scale = surrogate_no_target_scaling.predict_fidelity(single_sample)
    print(f"No target scaling: Cost={cost_single_no_scale[0]:.2f}, Fidelity={fidelity_single_no_scale[0]:.2f}")

    cost_single_scale = surrogate_with_target_scaling.predict_cost(single_sample)
    fidelity_single_scale = surrogate_with_target_scaling.predict_fidelity(single_sample)
    print(f"With target scaling: Cost={cost_single_scale[0]:.2f}, Fidelity={fidelity_single_scale[0]:.2f}")

    # Example of trying to predict before training
    fresh_surrogate = BasicSklearnSurrogate()
    try:
        print("\n--- Attempting prediction before training (should fail) ---")
        fresh_surrogate.predict_cost(X_test)
    except RuntimeError as e:
        print(f"Caught expected error: {e}")
