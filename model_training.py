from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


print("Starting model training...")

def train_polynomial_regression(X_train, y_train, degree):
    """
    Train a polynomial regression model of given degree
    """
    print("\n[TRAIN] Initializing PolynomialFeatures with degree =", degree)
    poly = PolynomialFeatures(degree=degree)

    print("[TRAIN] Transforming X_train to polynomial features...")
    X_train_poly = poly.fit_transform(X_train)
    print("[TRAIN] X_train_poly shape:", X_train_poly.shape)

    print("[TRAIN] Initializing LinearRegression model...")
    model = LinearRegression()

    print("[TRAIN] Fitting model...")
    model.fit(X_train_poly, y_train)

    print("[TRAIN] Training complete.")
    print("[TRAIN] Model coefficients:", model.coef_)
    print("[TRAIN] Model intercept:", model.intercept_)

    return model, poly
    

def evaluate_model(model, poly, X, y):
    """
    Evaluate model using Mean Squared Error
    """
    print("\n[EVAL] Transforming input data to polynomial features...")
    X_poly = poly.transform(X)
    print("[EVAL] X_poly shape:", X_poly.shape)

    print("[EVAL] Making predictions...")
    predictions = model.predict(X_poly)
    print("[EVAL] Predictions:", predictions)

    print("[EVAL] Calculating Mean Squared Error...")
    mse = mean_squared_error(y, predictions)
    print("[EVAL] MSE:", mse)

    print("[EVAL] Evaluation complete.\n")

    return mse, predictions
