import numpy as np

class ScratchLinearRegression:
    def __init__(self):
        self.theta = None

    def fit(self, features, target):
        # Add a column of ones to the features to account for the bias term
        bias = np.ones((features.shape[0], 1))
        enhanced_features = np.concatenate((bias, features), axis=1)
#calculate the transpose of enhanced feature matrix
        transposed_features = enhanced_features.T
        # Compute the inverse of the matrix (XᵀX)
        inverse_matrix = np.linalg.inv(np.dot(transposed_features, enhanced_features))
        pseudo_inverse = np.dot(inverse_matrix, transposed_features)
        self.theta = np.dot(pseudo_inverse, target)

    def predict(self, data):
        # Add a column of ones to the data to account for the bias term
        bias = np.ones((data.shape[0], 1))
        augmented_data = np.concatenate((bias, data), axis=1)
        # we get the predict value by multiplying the augmented data with the model parameters (theta)
        estimations = np.matmul(augmented_data, self.theta)
        return estimations
#the below functions were required for cross validatin tool to work (cross-val_score)
    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
