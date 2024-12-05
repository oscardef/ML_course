# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """Calculate the least squares solution.
       Returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape (D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # Compute optimal weights using the normal equation
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    
    # Calculate mean squared error
    e = y - tx @ w
    mse = (e @ e) / (2 * len(y))
    
    return w, mse
