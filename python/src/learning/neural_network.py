import torch
from torch import nn
import numpy as np
    
# Define NN class with scaling
class Scaled_MLP(nn.Module):

    def __init__(self,
                 nin,
                 nout,
                 feature_mean = None,
                 feature_std = None,
                 label_mean = None,
                 label_std = None,
                 activation='tanh',
                 n_layers=1,
                 n_neurons=30,
                 device=torch.device('cpu'),
                 dtype=torch.float32):
        super(Scaled_MLP, self).__init__()
        
        self.device = device
        self.dtype = dtype
        
        self.nin = nin
        self.nout = nout

        if feature_mean is None:
            self.feature_mean = torch.zeros(0,nin)
        else:
            self.feature_mean = torch.tensor(feature_mean,dtype=dtype,device=device).view(1,-1)
        
        if feature_std is None:
            self.feature_std = torch.ones(1,nin)
        else:
            self.feature_std = torch.tensor(feature_std,dtype=dtype,device=device).view(1,-1)

        if label_mean is None:
            self.label_mean = torch.zeros(0,nout)
        else:
            self.label_mean = torch.tensor(label_mean,dtype=dtype,device=device).view(1,-1)
        
        if label_std is None:
            self.label_std = torch.ones(1,nout)
        else:
            self.label_std = torch.tensor(label_std,dtype=dtype,device=device).view(1,-1)
        
        self.activation = activation
        self.n_layers = n_layers
        if isinstance(n_neurons,int):
            self.n_neurons = [self.nin]+([n_neurons]*n_layers)
        elif hasattr(n_neurons, "__len__") and len(n_neurons)==n_layers and all(isinstance(x,(int,np.generic)) for x in n_neurons):
            self.n_neurons = [self.nin]+list(n_neurons)
        else:
            raise Exception(f'Parameter n_neurons ill defined: expected int or array of length {n_layers}, got {n_neurons}!')
        
        activations = {'gelu':nn.GELU(),'sigmoid':nn.Sigmoid(),'tanh':nn.Tanh()}
        try:
            self.actfun = activations[self.activation]
        except Exception as e:
            raise Exception(f'Unknown activation function! Supported activations: [relu,sigmoid,tanh]; received: {self.activation}.')

        self.fclist = torch.nn.ModuleList([nn.Sequential(nn.Linear(self.n_neurons[i], self.n_neurons[i+1],device=self.device,dtype=self.dtype),self.actfun) for i in range(self.n_layers)])
        self.out = nn.Linear(self.n_neurons[-1], self.nout,device=self.device,dtype=self.dtype)

    def forward(self, x):
        x = (x - self.feature_mean)/self.feature_std
        for i in range(self.n_layers):
            x = self.fclist[i](x)
        x = self.out(x)
        return x

 # State Seperated MLP
class Scaled_Seperated_MLP(nn.Module):
    def __init__(self,
                 nin,
                 nout,
                 feature_mean = None,
                 feature_std = None,
                 label_mean = None,
                 label_std = None,
                 activation='tanh',
                 n_layers=1,
                 n_neurons=30,
                 device=torch.device('cpu'),
                 dtype=torch.float32):
        super(Scaled_Seperated_MLP, self).__init__()

        # Define Scaling
        if feature_mean is None:
            self.feature_mean = torch.zeros(0,nin)
        else:
            self.feature_mean = torch.tensor(feature_mean,dtype=dtype,device=device).view(1,-1)
        
        if feature_std is None:
            self.feature_std = torch.ones(1,nin)
        else:
            self.feature_std = torch.tensor(feature_std,dtype=dtype,device=device).view(1,-1)

        if label_mean is None:
            self.label_mean = torch.zeros(0,nout)
        else:
            self.label_mean = torch.tensor(label_mean,dtype=dtype,device=device).view(1,-1)
        
        if label_std is None:
            self.label_std = torch.ones(1,nout)
        else:
            self.label_std = torch.tensor(label_std,dtype=dtype,device=device).view(1,-1)
        
        self.nout = nout
        self.NNs = torch.nn.ModuleList([Scaled_MLP(nin,1,feature_mean,feature_std,label_mean,label_std,activation,n_layers,n_neurons,device,dtype) for _ in range(self.nout)])

    def forward(self, x):
        out = [NN(x) for NN in self.NNs]
        return torch.cat(out, dim=1)
    

# Ensamble NN
class Scaled_Ensemble_MLP(nn.Module):

    def __init__(self,
                 nin,
                 nout,
                 feature_mean = None,
                 feature_std = None,
                 label_mean = None,
                 label_std = None,
                 activation='tanh',
                 n_layers=1,
                 n_neurons=30,
                 n_models=3,
                 device=torch.device('cpu'),
                 dtype=torch.float32):
        super(Scaled_Ensemble_MLP, self).__init__()

        # Define Scaling
        if feature_mean is None:
            self.feature_mean = torch.zeros(0,nin)
        else:
            self.feature_mean = torch.tensor(feature_mean,dtype=dtype,device=device).view(1,-1)
        
        if feature_std is None:
            self.feature_std = torch.ones(1,nin)
        else:
            self.feature_std = torch.tensor(feature_std,dtype=dtype,device=device).view(1,-1)

        if label_mean is None:
            self.label_mean = torch.zeros(0,nout)
        else:
            self.label_mean = torch.tensor(label_mean,dtype=dtype,device=device).view(1,-1)
        
        if label_std is None:
            self.label_std = torch.ones(1,nout)
        else:
            self.label_std = torch.tensor(label_std,dtype=dtype,device=device).view(1,-1)
        
        self.nmodels = n_models
        self.NNs = torch.nn.ModuleList([Scaled_MLP(nin,nout,feature_mean,feature_std,label_mean,label_std,activation,n_layers,n_neurons,device,dtype) for _ in len(self.nmodels)])

    def forward(self, x):
        out = [mlp(x) for mlp in self.mlps]
        return torch.mean(torch.stack(out, dim=0), dim=0)

   
# Define NN class with scaling
class Scaled_Constraint_MLP(nn.Module):

    def __init__(self,
                 nin,
                 nout,
                 feature_mean = None,
                 feature_std = None,
                 label_mean = None,
                 label_std = None,
                 out_constraints_high = None,
                 out_constraints_low = None,
                 activation='tanh',
                 n_layers=1,
                 n_neurons=30,
                 device=torch.device('cpu'),
                 dtype=torch.float32):
        super(Scaled_Constraint_MLP, self).__init__()
        
        self.device = device
        self.dtype = dtype
        
        self.nin = nin
        self.nout = nout

        if feature_mean is None:
            self.feature_mean = torch.zeros(0,nin)
        else:
            self.feature_mean = torch.tensor(feature_mean,dtype=dtype,device=device).view(1,-1)
        
        if feature_std is None:
            self.feature_std = torch.ones(1,nin)
        else:
            self.feature_std = torch.tensor(feature_std,dtype=dtype,device=device).view(1,-1)

        if label_mean is None:
            self.label_mean = torch.zeros(0,nout)
        else:
            self.label_mean = torch.tensor(label_mean,dtype=dtype,device=device).view(1,-1)
        
        if label_std is None:
            self.label_std = torch.ones(1,nout)
        else:
            self.label_std = torch.tensor(label_std,dtype=dtype,device=device).view(1,-1)
        
        self.activation = activation
        self.n_layers = n_layers
        if isinstance(n_neurons,int):
            self.n_neurons = [self.nin]+([n_neurons]*n_layers)
        elif hasattr(n_neurons, "__len__") and len(n_neurons)==n_layers and all(isinstance(x,(int,np.generic)) for x in n_neurons):
            self.n_neurons = [self.nin]+list(n_neurons)
        else:
            raise Exception(f'Parameter n_neurons ill defined: expected int or array of length {n_layers}, got {n_neurons}!')
        
        activations = {'gelu':nn.GELU(),'sigmoid':nn.Sigmoid(),'tanh':nn.Tanh()}
        try:
            self.actfun = activations[self.activation]
        except Exception as e:
            raise Exception(f'Unknown activation function! Supported activations: [relu,sigmoid,tanh]; received: {self.activation}.')

        self.fclist = torch.nn.ModuleList([nn.Sequential(nn.Linear(self.n_neurons[i], self.n_neurons[i+1],device=self.device,dtype=self.dtype),self.actfun) for i in range(self.n_layers)])
        self.out = nn.Linear(self.n_neurons[-1], self.nout,device=self.device,dtype=self.dtype)

        if out_constraints_high != None and out_constraints_low != None:
            self.const_high = (torch.tensor(out_constraints_high,dtype=self.dtype) - self.label_mean)/self.label_std
            self.const_low = (torch.tensor(out_constraints_low,dtype=self.dtype) - self.label_mean)/self.label_std
        else:
            self.const_high = None
            self.const_low = None

    def forward(self, x):
        x = (x - self.feature_mean)/self.feature_std
        for i in range(self.n_layers):
            x = self.fclist[i](x)
        x = self.out(x)
        if self.const_high != None and self.const_low != None:
            x = self.const_low + (self.const_high - self.const_low) * torch.sigmoid(x)
        return x