# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:30:15 2024

@author: rose
"""

import numpy as np

def datasample(array, n, replace=False):
    """
    Randomly samples `n` elements from `array` and returns the sampled elements
    along with their corresponding indices. It replicates the "Datasample()
    function from matlab" 

    Parameters:
    array (numpy.ndarray): The input array to sample from.
    n (int): Number of samples to draw.
    replace (bool): Whether to sample with replacement (default is False).

    Returns:
    sampled_elements (numpy.ndarray): The sampled elements.
    sampled_indices (numpy.ndarray): The indices of the sampled elements.
    """
    sampled_indices = np.random.choice(len(array), size=n, replace=replace)
    sampled_elements = array[sampled_indices]
    return sampled_elements, sampled_indices