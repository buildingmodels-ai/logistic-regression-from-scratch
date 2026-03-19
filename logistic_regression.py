import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import accuracy_score


class LogisticRegression:
    """
    Logistic Regression implemented from scratch.
    Uses gradient descent with cross-entropy loss.
    """

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cross_entropy_loss(self, y_true, y_pred):
        m = len(y_true)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -1/m * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for i in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            loss = self.cross_entropy_loss(y, y_pred)
            self.losses.append(loss)

            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return (y_pred >= 0.5).astype(int)


if __name__ == "__main__":

    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train custom model
    custom_model = LogisticRegression(learning_rate=0.1, iterations=1000)
    custom_model.fit(X_train, y_train)

    # Train sklearn model (baseline comparison)
    sklearn_model = SklearnLR()
    sklearn_model.fit(X_train, y_train)

    # Evaluate
    y_pred_custom = custom_model.predict(X_test)
    y_pred_sklearn = sklearn_model.predict(X_test)

    acc_custom = accuracy_score(y_test, y_pred_custom)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

    print("\n=== Results ===")
    print(f"Custom Model Accuracy: {acc_custom:.4f}")
    print(f"Sklearn Model Accuracy: {acc_sklearn:.4f}")

    # Plot loss curve
    plt.plot(custom_model.losses)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True)
    plt.savefig("training_loss.png", dpi=300)
