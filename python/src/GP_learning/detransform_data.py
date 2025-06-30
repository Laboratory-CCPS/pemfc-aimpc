# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:16:01 2024

@author: rose
"""

import numpy as np

def detransform_data(Data_transformed, transform_info):
    """
    Reverses the transformation applied to the data based on the provided transform_info.
    
    Parameters:
    Data_transformed (numpy.ndarray): The transformed data to be reversed.
    transform_info (dict): A dictionary containing the information needed to reverse the transformation.
    
    Returns:
    Data (numpy.ndarray): The original data before transformation.
    """
    
    if transform_info['method'] == 'none':  # No transformation
        Data = Data_transformed
    
    elif transform_info['method'] == 'center':  # Reversing centering (zero mean)
        Data = Data_transformed + transform_info['mean_data']
    
    elif transform_info['method'] == 'zscore':  # Reversing z-score normalization
        mean_data = transform_info['mean_data']
        std_data = transform_info['std_data']
        Data = Data_transformed * std_data + mean_data
    
    elif transform_info['method'] == 'rescale':  # Reversing rescaling
        min_data = transform_info['min_data']
        max_data = transform_info['max_data']
        a = transform_info['lb']
        b = transform_info['ub']
        Data = ((Data_transformed - a) / (b - a)) * (max_data - min_data) + min_data
    
    return Data