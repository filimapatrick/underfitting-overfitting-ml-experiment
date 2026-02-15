import numpy as np
from sklearn.model_selection import train_test_split


def generate_nonlinear_data(n_samples=100, noise_std=0.2, random_state=42):
    """
    Generate synthetic nonlinear data:
    y = sin(x) + noise
    """

    np.random.seed(random_state)

    X = np.linspace(0, 10, n_samples)
    X = X.reshape(-1, 1)

    y = np.sin(X) + np.random.normal(0, noise_std, X.shape)

    # Print generated data
    print("Generated X:\n", X[:10], "...")  # show first 10 for brevity
    print("Generated y:\n", y[:10], "...\n")  # show first 10 for brevity

    return X, y


def split_data(X, y, test_size=0.3, random_state=42):
    """
    Split dataset into training and testing sets
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Print the split results
    print("X_train:\n", X_train[:10], "...")
    print("X_test:\n", X_test[:10], "...")
    print("y_train:\n", y_train[:10], "...")
    print("y_test:\n", y_test[:10], "...\n")

    return X_train, X_test, y_train, y_test


# Example usage
if __name__ == "__main__":
    X, y = generate_nonlinear_data(n_samples=20)  # smaller sample to see output
    split_data(X, y)
