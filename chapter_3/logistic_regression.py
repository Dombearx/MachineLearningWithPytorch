import os
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))  # from chapter_2 -> project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from chapter_2.adaline import AdalineSGD


class LogisticRegressionSGD(AdalineSGD):
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=42, shuffle=True):
        super().__init__(learning_rate, n_iterations, random_state, shuffle)

    def activation(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(np.clip(-X, -250, 250)))  # Sigmoid function
    
    def calculate_loss(self, y: np.ndarray, output: np.ndarray) -> float:
        loss = -y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output))
        return loss / y.shape[0]
