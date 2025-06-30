import itertools

from tqdm import tqdm
from torch.optim import Adam
from torch.nn import MSELoss

from .neural_network import Scaled_Constraint_MLP
from .training import TrainLoopScaled,ValidLoopScaled


# Grid Search function for tuning
def hyperparameter_grid_search(trainloader,
                               validloader,
                               feature_scaler,
                               label_scaler,
                               neuron_values:list=[100],
                               layer_values:list=[3],
                               out_constraints_high:list=[500,250],
                               out_constraints_low:list=[0,50],
                               activation_values:list=['tanh'],
                               lr_values:list=[1e-3],
                               epoch_values:list=[1000]):
 
    # Define parameter grid
    param_grid = {
        'n_neurons': neuron_values,  # Try different architectures
        'n_layers' : layer_values,
        'activations' : activation_values,
        'lr': lr_values,  # Try different learning rates
        'epochs': epoch_values  # Number of epochs to train
    }
    combinations = list(itertools.product(*param_grid.values()))
    combinations_dicts = [dict(zip(param_grid.keys(), values)) for values in combinations]

    best_r2score = -1e10
    best_params = {}

    for combi in tqdm(combinations_dicts,desc='Trials'):
                # Initialize model, optimizer, and loss function
                model = Scaled_Constraint_MLP(nin = 4,
                                              nout = 2,
                                              feature_mean=feature_scaler.mean_,
                                              feature_std=feature_scaler.scale_,
                                              label_mean=label_scaler.mean_,
                                              label_std=label_scaler.scale_,
                                              out_constraints_low=out_constraints_low,
                                              out_constraints_high=out_constraints_high,
                                              activation = combi['activations'],
                                              n_layers = combi['n_layers'],
                                              n_neurons = combi['n_neurons'])
                optimizer = Adam(model.parameters(), lr=combi['lr'])
                lossfun = MSELoss()

                # Train the model
                for epoch in range(combi['epochs']):
                    _ = TrainLoopScaled(model,trainloader,lossfun,optimizer)
                _,r2 = ValidLoopScaled(model,validloader,lossfun)
                if r2 > best_r2score:
                    best_r2score = r2
                    best_params = {'n_neurons': combi['n_neurons'],
                                   'n_layers' : combi['n_layers'],
                                   'activations' : combi['activations'],
                                   'lr': combi['lr'], 
                                   'epochs': combi['epochs']}

    return best_params, best_r2score