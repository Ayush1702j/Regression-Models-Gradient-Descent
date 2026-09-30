"""Regression model definitions and custom Gradient Descent implementation."""

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class GradientDescentResult:
    theta: np.ndarray
    cost_history: list[float]
    iterations: int


class LinearRegressionGD:
    """Multivariate Linear Regression trained using batch Gradient Descent."""

    def __init__(self, learning_rate=0.05, n_iterations=5000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.cost_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        Xb = np.c_[np.ones(X.shape[0]), X]
        self.theta = np.zeros(Xb.shape[1], dtype=float)
        self.cost_history = []
        m = len(y)

        for _ in range(self.n_iterations):
            predictions = Xb @ self.theta
            errors = predictions - y
            cost = float(np.mean(errors ** 2) / 2.0)
            self.cost_history.append(cost)
            gradient = (Xb.T @ errors) / m
            self.theta -= self.learning_rate * gradient

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xb = np.c_[np.ones(X.shape[0]), X]
        return Xb @ self.theta


def build_models():
    """Return the classical regression models used in the experiment."""
    return {
        "Linear Regression": LinearRegression(),
        "Polynomial Regression (Degree 2)": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("linear", LinearRegression()),
        ]),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=20000),
    }


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }
