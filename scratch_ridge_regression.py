import numpy as np

class ScratchRidgeRegression:
    def __init__(self, alpha):
        self.alpha = alpha
        self.theta = None

    def fit(self, features, target):
        bias = np.ones((features.shape[0], 1))  # Add bias term
        enhanced_features = np.concatenate((bias, features), axis=1)

        # Ridge regression closed-form solution: θ = (XᵀX + λI)⁻¹Xᵀy
        n_features = enhanced_features.shape[1]
        identity_matrix = np.eye(n_features)  # Identity matrix of size (n_features x n_features)
        identity_matrix[0, 0] = 0  # Do not regularize the bias term

        transposed_features = enhanced_features.T
        regularization_term = self.alpha * identity_matrix

        # Compute (XᵀX + λI) and its inverse
        to_invert = np.dot(transposed_features, enhanced_features) + regularization_term
        inverse_matrix = np.linalg.inv(to_invert)

        # Compute the final weights
        self.theta = np.dot(np.dot(inverse_matrix, transposed_features), target)

    def predict(self, data):
        bias = np.ones((data.shape[0], 1))
        augmented_data = np.concatenate((bias, data), axis=1)
        estimations = np.dot(augmented_data, self.theta)
        return estimations

    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
