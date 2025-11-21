from typing import Any


import numpy as np

class Adaline:
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=42):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        random_state = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        self.weights = random_state.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias = np.float64(0.0)
        self.losses = []

        for _ in range(self.n_iterations):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.weights += self.learning_rate * 2.0 * X.T.dot(errors) / n_samples
            self.bias += self.learning_rate * 2.0 * errors.mean()
            loss = self.calculate_loss(y, output)
            self.losses.append(loss)

    def calculate_loss(self, y: np.ndarray, output: np.ndarray) -> float:
        errors = y - output
        loss = (errors**2).mean()
        return loss

    def activation(self, X: np.ndarray) -> np.ndarray:
        return X

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.net_input(X) >= 0.5, 1, 0)


class AdalineSGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=42, shuffle=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.random_state = random_state
        self.shuffle = shuffle
        self.weights_initialized = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.init_weights(n_features)
        self.losses = []

        for _ in range(self.n_iterations):
            epoch_losses = []
            if self.shuffle:
                X, y = self.shuffle_data(X, y)
            for xi, target in zip(X, y):
                loss = self.update_weights(xi, target)
                epoch_losses.append(loss)
            mean_epoch_loss = np.mean(epoch_losses)
            self.losses.append(mean_epoch_loss)

    def parital_fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        if not self.weights_initialized: 
            self.init_weights(n_features)    
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self.update_weights(xi, target)
        else:
            self.update_weights(X, y)
        
    def activation(self, X: np.ndarray) -> np.ndarray:
        return X

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.net_input(X) >= 0.5, 1, 0)

    def init_weights(self, n_features: int):
        self.random_state = np.random.RandomState(self.random_state)
        self.weights = self.random_state.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias = np.float64(0.0)
        self.weights_initialized = True
 
    def shuffle_data(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = self.random_state.permutation(len(y))
        return X[r], y[r]

    def update_weights(self, xi: np.ndarray, target: np.ndarray) -> float:
        net_result = self.net_input(xi)
        output = self.activation(net_result)
        error = target - output
        self.weights += self.learning_rate * 2.0 * xi * error
        self.bias += self.learning_rate * 2.0 * error
        loss = error**2
        return loss