import matplotlib.pyplot as plt
import numpy as np
import os


def plot_regression_curve(X, y, model, poly, degree, save_path=None):
    """
    Plot regression curve for a specific polynomial degree
    """

    plt.figure(figsize=(6, 4))
    plt.scatter(X, y, alpha=0.5)

    X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot = model.predict(X_plot_poly)

    plt.plot(X_plot, y_plot, color='red')
    plt.title(f"Polynomial Degree {degree}")
    plt.xlabel("X")
    plt.ylabel("y")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()


def plot_error_curve(degrees, train_errors, test_errors, save_path=None):
    """
    Plot training vs testing error curve
    """

    plt.figure(figsize=(8, 5))
    plt.plot(degrees, train_errors, label="Training Error")
    plt.plot(degrees, test_errors, label="Testing Error")

    plt.xlabel("Polynomial Degree")
    plt.ylabel("Mean Squared Error")
    plt.title("Bias-Variance Tradeoff")
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()
