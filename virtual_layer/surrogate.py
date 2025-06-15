import numpy as np

class SurrogateModel:
    """Learned surrogate for primitive cost and fidelity."""
    def __init__(self, primitive_name: str):
        self.name = primitive_name
        self.model = None  # placeholder for regression model
        self.trained = False

    def train(self, data: np.ndarray):
        """Train surrogate on execution logs."""
        # TODO: implement training (e.g., Bayesian linear regression)
        self.trained = True

    def predict_cost(self, latent: np.ndarray) -> float:
        if not self.trained:
            raise RuntimeError("Surrogate not trained")
        # TODO: predict cost from latent
        return 0.0

    def predict_fidelity(self, latent: np.ndarray) -> float:
        if not self.trained:
            return 1.0
        # TODO: predict fidelity from latent
        return 1.0
