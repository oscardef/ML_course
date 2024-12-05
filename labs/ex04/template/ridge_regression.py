# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import compute_mse


def ridge_regression(y, tx, lambda_):
    """
    Perform ridge regression using normal equations.

    Parameters:
    - y: numpy array, shape (N,), the output values.
    - tx: numpy array, shape (N,D), the input features.
    - lambda_: float, the regularization parameter.

    Returns:
    - w: numpy array, the optimal weight vector.
    - loss: float, the final loss value (mean squared error, excluding the penalty term).
    """

    tx_transposed = tx.T

    a = tx_transposed @ tx + 2 * tx.shape[0] * lambda_ * np.identity(n=tx.shape[1])
    b = tx_transposed @ y
    w = np.linalg.solve(a=a, b=b)

    loss = compute_mse(y=y, tx=tx, w=w)

    return w, loss
