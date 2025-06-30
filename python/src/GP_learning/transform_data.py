# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:10:37 2024

@author: rose
"""

import numpy as np

def transform_data(Data, method, transform_info=None):
    """
    Transforms the input data according to the specified method.
    If transform_info is provided, the transformation is based on that information.
    
    Parameters:
    Data (numpy.ndarray): The input data (shape should be [n_samples, n_features]).
    method (str): The transformation method ('none', 'center', 'zscore', 'rescale').
    transform_info (dict, optional): A dictionary with precomputed transformation information (default is None).
    
    Returns:
    Data_transformed (numpy.ndarray): The transformed data.
    transform_info (dict): Information used for the transformation (e.g., mean, std, min, max).
    """
    if method == 'none':  # No transformation
        Data_transformed = Data
        transform_info = {'method': 'none'}

    elif method == 'center':  # Center data (zero mean)
        mean_data = np.mean(Data, axis=0)
        Data_transformed = Data - mean_data
        transform_info = {'method': 'center', 'mean_data': mean_data}

    elif method == 'zscore':  # Z-score normalization (zero mean, unit variance)
        mean_data = np.mean(Data, axis=0)
        std_data = np.std(Data, axis=0)
        Data_transformed = (Data - mean_data) / std_data
        transform_info = {'method': 'zscore', 'mean_data': mean_data, 'std_data': std_data}

    elif method == 'rescale':  # Rescale data between -1 and 1
        min_data = np.min(Data, axis=0)
        max_data = np.max(Data, axis=0)
        a, b = -1, 1
        Data_transformed = a + ((Data - min_data) / (max_data - min_data)) * (b - a)
        transform_info = {'method': 'rescale', 'min_data': min_data, 'max_data': max_data, 'lb': a, 'ub': b}

    elif transform_info is not None:  # Transform using provided rules
        if transform_info['method'] == 'none':  # No transformation
            Data_transformed = Data

        elif transform_info['method'] == 'center':  # Center data (zero mean)
            Data_transformed = Data - transform_info['mean_data']

        elif transform_info['method'] == 'zscore':  # Z-score normalization (zero mean, unit variance)
            Data_transformed = (Data - transform_info['mean_data']) / transform_info['std_data']

        elif transform_info['method'] == 'rescale':  # Rescale data using provided min/max
            min_data = transform_info['min_data']
            max_data = transform_info['max_data']
            a, b = transform_info['lb'], transform_info['ub']
            Data_transformed = a + ((Data - min_data) / (max_data - min_data)) * (b - a)

    return Data_transformed, transform_info