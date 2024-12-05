# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N, degree+1)

    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    # Initialize an output array of shape (N, degree+1)
    poly = np.zeros((x.shape[0], degree + 1))
    
    # Fill each column with x raised to the power of the corresponding degree
    for j in range(degree + 1):
        poly[:, j] = x ** j
        
    return poly