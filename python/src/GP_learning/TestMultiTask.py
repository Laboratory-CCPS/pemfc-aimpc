# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:16:49 2024

@author: rose
AIEmbedded_TestmultiTaskGP
"""
'imports'
import scipy.io as sio
import numpy as np
from transform_data import transform_data
from detransform_data import detransform_data
from datasample import datasample
import torch
import gpytorch

'load Data'
D = sio.loadmat('Data_FC_10000_2.mat')
D = D['Data']
Data_X = D[:,0:2]
Data_Y = D[:,3]
n_D = len(Data_X)
n_x = 5
n_y = 2
# get X and Y only

X = np.array(  [ np.concatenate( (Data_X[i,0], Data_X[i,1]) )  for i in range(n_D)  ] ).reshape((n_D,n_x)) 
Y = np.array([np.array( (Data_Y[i][0,0], Data_Y[i][1,0]) ) for i in range(n_D)] )
'scale Data'
X_scale, trans_info_X = transform_data(X, 'zscore')
Y_scale, trans_info_y = transform_data(Y,'zscore')
# X_detransformed = detransform_data(X_scale, trans_info_X) # check if same as X

## to reduce comp demand - reduce number of data

X_samp, idx_samp = datasample(X, n=len(X_scale), replace=False)
Y_samp = Y_scale[idx_samp,:]
'data to torch'
X_GP = torch.tensor(X_samp,dtype=torch.float32)
Y_GP = torch.tensor(Y_samp,dtype=torch.float32)
train_x = X_GP
train_y = Y_GP

'define multi task gp'


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=n_x), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
model = MultitaskGPModel(train_x, train_y, likelihood)




'train gp hyperparas'

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iterations = 100

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()



'save GP'

# torch.save(model.state_dict(), 'model_state2811.pth')

'test GP'
model.eval()
likelihood.eval()

'endofscricp'
endofscript = 1



