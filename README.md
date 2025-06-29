# PEM-Fuelcell AI-MPC

This repository contains code developed within the KI-Embedded project, focusing on model predictive control (MPC) for fuel cell systems. It includes implementations in both MATLAB and Python for modeling, control, and machine learning-based approximation of predictive controllers.

## Overview

The project explores the integration of AI techniques into MPC for embedded control of PEM fuel cells. It covers:

- **Fuel Cell Modeling**
- **Model Predictive Control (MPC)**
- **Observer Design**
- **Gaussian Process (GP) Regression**
- **Neural Network-based MPC Approximation**

## Contents

### MATLAB
- `model/`: PEM fuel cell models used in control tasks.
- `mpc/`: Classical MPC design using CasADi (via MATLAB interface).
- `observer/`: Observer design (e.g., Kalman filters, output feedback).
- `gp/`: Gaussian Process models using the GPML MATLAB toolbox.

### Python
- `model/`: PEM fuel cell models used in control tasks (equivalent to Matlab version)
- - `mpc/`: Classical MPC design using CasADi (via Python interface).
- `nn_mpc/`: Neural Network-based MPC approximation using PyTorch.
- `utils/`: Helper scripts for training, evaluation, and simulation.

## Requirements

### MATLAB
- MATLAB R2022b or newer
- CasADi for MATLAB
- GPML Toolbox (https://gaussianprocess.org/gpml/code/matlab/doc/)

### Python
- Python 3.9+
- `Casadi`
- `torch`
- `casadi`
- `numpy`, `scipy`, `matplotlib`
