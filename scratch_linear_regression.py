import numpy as np

class ScratchLinearRegression:
    def __init__(self):
        self.theta = None

    def fit(self, features, target):
        bias = np.ones((features.shape[0], 1))
        enhanced_features = np.concatenate((bias, features), axis=1)

        transposed_features = enhanced_features.T
        inverse_matrix = np.linalg.inv(np.dot(transposed_features, enhanced_features))
        pseudo_inverse = np.dot(inverse_matrix, transposed_features)
        self.theta = np.dot(pseudo_inverse, target)

    def predict(self, data):
        bias = np.ones((data.shape[0], 1))
        augmented_data = np.concatenate((bias, data), axis=1)
        estimations = np.matmul(augmented_data, self.theta)
        return estimations

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
