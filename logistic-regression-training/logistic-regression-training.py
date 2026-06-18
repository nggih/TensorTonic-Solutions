import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)

    m, n = X.shape 
    w = np.zeros(n)
    b = 0.0

    for step in range(steps):
        z = X @ w + b 
        y_pred = _sigmoid(z)

        eps = 1e-15
        y_pred_safe = np.clip(y_pred, eps, 1- eps)

        loss = -np.mean(y * np.log(y_pred_safe) + (1 - y) * np.log(1 - y_pred_safe))

        error = y_pred - y 
        dw = (1/ m) * X.T @ error 
        db = (1/ m) * np.sum(error)

        w -= lr * dw 
        b -= lr * db 

    return (w, b)