import numpy as np

class ScratchPolynomialRegression:
    def __init__(self, degree):
        self.degree = degree
        self.theta = None

    def fit(self, features, target):
        # changinf the featurs interms of polynomial terms
        polynomial_features = self._polynomial_features(features)
        
        # we add the bias term
        bias = np.ones((polynomial_features.shape[0], 1))
        enhanced_features = np.concatenate((bias, polynomial_features), axis=1)
        
        # we know Normal equation is: θ = (XᵀX)⁻¹Xᵀy
        transposed_features = enhanced_features.T
        to_invert = np.dot(transposed_features, enhanced_features)
        inverse_matrix = np.linalg.inv(to_invert)
        
        # then the final weights would be
        self.theta = np.dot(np.dot(inverse_matrix, transposed_features), target)

    def predict(self, data):
        polynomial_data = self._polynomial_features(data)
        bias = np.ones((polynomial_data.shape[0], 1))
        augmented_data = np.concatenate((bias, polynomial_data), axis=1)
        estimations = np.dot(augmented_data, self.theta)
        return estimations
    
    def _polynomial_features(self, features):
        # Generating the polynomial featurs
        poly_features = np.ones((features.shape[0], 1))
        for d in range(1, self.degree + 1):
            poly_features = np.concatenate((poly_features, features ** d), axis=1)
        return poly_features

    def get_params(self, deep=True):
        return {"degree": self.degree}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
