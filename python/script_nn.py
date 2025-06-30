# %% Imports
import torch
import optuna
import tqdm

import pandas as pd
import matplotlib.pyplot as plt
import optuna.visualization as vis
import numpy as np

from tqdm import trange
from timeit import default_timer

from src.learning.dataloader import *
from src.learning.neural_network import *
from src.learning.training import *
from src.learning.tuning import hyperparameter_grid_search
from src.model_suh import model_suh
from src.model_suh.conversions import rpm2rad
from src.mpc import mpc_problem, mpc_solver
from src.ode_modelling import simresult
from src.ode_modelling.discrete_model import discretize_model
from src.ode_modelling.ode_model import get_model
from src.ode_modelling.simresult import plot_sim_results
from src.ode_modelling.simulator import Simulator, sim_nl
from src.learning.utils import data_as_csv


# %% Flags
tune = None
train_with_load = True

# %% Dataset
batchsize = int(512/2)
workers = 0
valid_fraction = 0.2
const_high = [500,250]
const_low = [0,50]
if train_with_load:
    fnames = ['p_O2','p_N2','w_cp','p_sm','I_load']
else:
    fnames = ['p_O2','p_N2','w_cp','p_sm']
lnames = ['I_st','v_cm']
datapath = '.\data\mpc_data_vcm_min50_only_steps.csv' #'.\data\mpc_data_vcm_min50_only_steps.csv' #'.\data\mpc_data_vcm_min50_only_steps.csv' #'.\data\mpc_data_50.csv'

# %% Hyperparameter tuning via gridsearch or train once
if tune=='grid':
    print('Tuning via gridsearch.')

    train_loader,valid_loader,feature_scaler,label_scaler = get_dataset_and_scaler(pd.read_csv(datapath),
                                                                                    valid_frac=valid_fraction,
                                                                                    feature_columns=fnames,
                                                                                    label_columns=lnames,
                                                                                    batch_size=batchsize,
                                                                                    num_workers=workers)

    best_params, best_score = hyperparameter_grid_search(train_loader,
                                                         valid_loader,
                                                         feature_scaler=feature_scaler,
                                                         label_scaler=label_scaler,
                                                         neuron_values=[10,50,100],
                                                         layer_values=[2,5,10],
                                                         activation_values=['gelu','tanh','sigmoid'],
                                                         lr_values=[1e-1,1e-3,1e-5],
                                                         epoch_values=[500,1000,4000])

    print("Best Hyperparameters:", best_params)
    print("Best Score:", best_score)

elif tune=='gauss':
    print('Tuning via Bayesian optimization using Gaussian process for sampling.')

    num_trials = 5

    def objective(trial):
        # Suggest hyperparameters
        n_layers = trial.suggest_int("n_layers", 3, 6)
        n_neurons = trial.suggest_int("n_neurons",50,150)
        activation = trial.suggest_categorical("activation_fn", ["gelu", "tanh", "sigmoid"])
        learning_rate = trial.suggest_float("lr", 1e-4, 1e-2)
        epochs = trial.suggest_int("epochs",2000, 4000)

        # Dataset and scaler
        train_loader,valid_loader,feature_scaler,label_scaler = get_dataset_and_scaler(pd.read_csv(datapath),
                                                                                       valid_frac=valid_fraction,
                                                                                       feature_columns=fnames,
                                                                                       label_columns=lnames,
                                                                                       batch_size=batchsize,
                                                                                       num_workers=workers)
        
        # Initialize model, optimizer, and loss function
        model = Scaled_Constraint_MLP(nin=len(fnames),
                                      nout=len(lnames),
                                      n_neurons=n_neurons,
                                      n_layers=n_layers,
                                      activation=activation,
                                      feature_mean=feature_scaler.mean_,
                                      feature_std=feature_scaler.scale_,
                                      label_mean=label_scaler.mean_,
                                      label_std=label_scaler.scale_)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        lossfunc = nn.MSELoss()

        # Train the model
        for epoch in range(epochs):
            _ = TrainLoopScaled(model, train_loader, lossfunc, optimizer)

        # Evaluate the model
        _, r2 = ValidLoopScaled(model,valid_loader,lossfunc)

        return r2

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction="maximize",sampler=sampler,study_name='Tune_Imitation')
    study.optimize(objective, n_trials=num_trials,show_progress_bar=True)
    
    best_params = study.best_params

    print("Best Hyperparameters:", study.best_params)
    print("Best R2:", study.best_value)

    fig = vis.plot_param_importances(study)
    fig.show()

else:
    print('Training model with fixed hyperparameters.')

    # Dataset and scaler
    train_loader,valid_loader,feature_scaler,label_scaler = get_dataset_and_scaler(pd.read_csv(datapath),
                                                                                   valid_frac=valid_fraction,
                                                                                   feature_columns=fnames,
                                                                                   label_columns=lnames,
                                                                                   batch_size=batchsize,
                                                                                   num_workers=workers)

    # NN
    num_inputs = len(fnames)
    num_outputs = len(lnames)
    activation = 'sigmoid'
    num_layer = 2
    num_neurons = 10

    nnmodel = Scaled_Constraint_MLP(nin=num_inputs,
                               nout=num_outputs,
                               n_neurons=num_neurons,
                               n_layers=num_layer,
                               activation=activation,
                               feature_mean=feature_scaler.mean_,
                               feature_std=feature_scaler.scale_,
                               label_mean=label_scaler.mean_,
                               label_std=label_scaler.scale_,
                               out_constraints_high=const_high,
                               out_constraints_low=const_low)

    #nnmodel = Scaled_ResNet(nin=num_inputs,
    #                        nout=num_outputs,
    #                        n_neurons=num_neurons,
    #                        n_blocks=num_layer,
    #                        activation=activation,
    #                        label_scaler=label_scaler,
    #                        feature_scaler=feature_scaler)

    #nnmodel = Scaled_Seperated_MLP(nin=num_inputs,
    #                               nout=num_outputs,
    #                               n_neurons=num_neurons,
    #                               n_layers=num_layer,
    #                               activation=activation,
    #                               feature_scaler=feature_scaler,
    #                               label_scaler=label_scaler)

    # Training
    num_epochs = 75
    learningrate = 0.0006

    lossfun = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(nnmodel.parameters(),lr=learningrate,weight_decay=1e-4)

    log_dict = {'train_loss': [0 for _ in range(num_epochs)], 'valid_loss': [0 for _ in range(num_epochs)], 'r2': [0 for _ in range(num_epochs)]}

    for epoch in trange(num_epochs):

        log_dict['train_loss'][epoch] = TrainLoopScaled(nnmodel,train_loader,lossfun,optimizer)
        log_dict['valid_loss'][epoch],log_dict['r2'][epoch] = ValidLoopScaled(nnmodel,valid_loader,lossfun)

    print(f'Training finished with final R2-score {log_dict["r2"][-1]}')

    # Plotting
    fig, ax = plt.subplots(2,1)
    ax[0].plot(range(num_epochs),log_dict['train_loss'])
    ax[0].plot(range(num_epochs),log_dict['valid_loss'])
    ax[0].legend(['train loss','validation loss'])
    ax[0].set_yscale('log')

    ax[1].plot(range(num_epochs),log_dict['r2'])
 
    torch.save({'nin' : num_inputs,
                'nout' : num_outputs,
                'feature_scaler' : feature_scaler,
                'label_scaler' : label_scaler,
                'activation': activation,
                'n_layers': num_layer,
                'n_neurons': num_neurons,
                'const_high': const_high,
                'const_low' : const_low,
                'model_state_dict': nnmodel.state_dict(),
                }, 'current_best_nn.pickle')

# %% Closed-loop test
if train_with_load:
        p = model_suh.params()
        model = get_model(model_suh.model, p)

        plot_spec = ("@states, lambda_O2, @inputs",)

        mpcproblem = mpc_problem.MpcProblem(model)

        mpcproblem.add_free_input("I_st", 0, np.inf, 100)
        mpcproblem.add_free_input("v_cm", 50, 250, 110)

        mpcproblem.add_constraint("p_O2", 0, np.inf)
        mpcproblem.add_constraint("p_N2", 0, np.inf)
        mpcproblem.add_constraint("p_sm", 0, np.inf)
        mpcproblem.add_constraint("w_cp", rpm2rad(20e3), rpm2rad(105e3))
        mpcproblem.add_constraint("lambda_O2", 1.5, 5, feedthrough=False)
        mpcproblem.add_constraint("lambda_O2", 1.5, 5, feedthrough=True)

        mpcproblem.add_quadratic_cost("I_st", ref="#I_load[]", weight_stage=5e-2)
        mpcproblem.add_quadratic_cost("lambda_O2", ref=2, weight_stage=1, weight_end=1)

        mpcproblem.add_quadratic_diffquot_cost(
            "v_cm", diff_quot=(-1, 1), ts_order=0, weight=1e-5
        )

        Ts_mpc = 0.025
        Npredict = 10

        future_I_load_known = False

        discretization = "rk1"
        substeps = 2
        scale_model = True
        use_lambda = True
        multiple_shooting = True

        sim_oversampling = 10
        Ts_sim = Ts_mpc / sim_oversampling
        p_sim = p.copy()
        #p_sim.J_cp = p.J_cp * 0.5
        sim_model = get_model(model_suh.model, p_sim)

        simulator = Simulator(sim_model, Ts_sim)

        [u_ref, x0] = model_suh.get_validation_input(Ts_mpc)

        def get_I_load(t: np.ndarray | float) -> np.ndarray | float:
            return u_ref[1, np.minimum(np.round(t / Ts_mpc).astype(int), u_ref.shape[1])]
        
        T_end = 29

        mpcsolver = mpc_solver.MpcSolver(
            mpcproblem,
            Ts_mpc,
            Npredict,
            discretization,
            substeps=substeps,
            scale_model=scale_model,
            multiple_shooting=multiple_shooting,
            solver_options={},
            plugin_options={},
            verbose=False,
        )

        u_init = mpcsolver.get_u_init()
        x_init = x0
        lam_g = None

        n_mpc_steps = int(np.floor(T_end / Ts_mpc))

        u_mpc = np.nan * np.zeros((len(mpcsolver.free_inputs), n_mpc_steps))
        u_fixed = np.nan * np.zeros((len(mpcsolver.fixed_inputs), n_mpc_steps))

        n_sim_steps = sim_oversampling * n_mpc_steps
        t_sim = np.arange(n_sim_steps + 1) * Ts_sim
        x_sim = np.nan * np.zeros((sim_model.states.numel(), n_sim_steps + 1))
        u_sim = np.nan * np.zeros((sim_model.inputs.numel(), n_sim_steps + 1))
        y_sim = np.nan * np.zeros((len(sim_model.y_names), n_sim_steps + 1))
        y_ft_sim = np.nan * np.zeros((len(sim_model.y_names), n_sim_steps + 1))

        t_calc = np.nan * np.zeros(n_sim_steps+1)

        x_sim[:, 0] = x0

        for i in tqdm.tqdm(range(n_mpc_steps)):
            t_step = (np.arange(Npredict + 1) + i) * Ts_mpc

            if future_I_load_known:
                I_load_step = get_I_load(t_step)
            else:
                I_load_step = get_I_load(t_step[0])

            t_mpc_start = default_timer() 
            step_result = mpcsolver.solve_step(
                x0=x_init, u_init=u_init, lam_g=lam_g, I_load=I_load_step
            )
            t_mpc_stop = default_timer() 

            # shift the states by one step
            # (The first state will be overwritten below by the result of the
            # simulation.)
            (x_init, u_init, lam_g) = mpcsolver.get_next_initial_values(step_result)

            u_mpc[:, i] = step_result["u_opt"][:, 0]
            u_fixed[:, i] = step_result["u_fixed"][:, 0]

            # simulate plant
            sim_idx = sim_oversampling * i
            u_sim_cur = mpcsolver.get_complete_input(
                step_result["u_opt"][:, [0]], step_result["u_fixed"][:, 0], 1
            )
            x_sim_cur = x_sim[:, sim_idx]

            sim_error = False

            for ii in range(sim_oversampling):
                try:
                    [x_sim_next, y_sim_next, y_sim_cur_ft] = simulator.sim_step(
                        x_sim_cur, u_sim_cur, feedthrough_start=True
                    )
                except RuntimeError:
                    sim_error = True
                    break

                u_sim[:, [sim_idx]] = u_sim_cur
                y_ft_sim[:, [sim_idx]] = y_sim_cur_ft
                t_calc[sim_idx] = t_mpc_stop - t_mpc_start

                x_sim[:, [sim_idx + 1]] = x_sim_next
                y_sim[:, [sim_idx + 1]] = y_sim_next

                x_sim_cur = x_sim_next

                sim_idx = sim_idx + 1

            if sim_error:
                break

            # use plant state as "true" state, i.e. use this as the initial state
            # for the next mpc step
            x_init[:, [0]] = x_sim_next

        mpc_result = simresult.SimResult.from_data(
            sim_model, t_sim, u_sim, x_sim, y_sim, y_ft_sim
        )

        mpc_result.constraints = mpcsolver.get_all_constraints()
        mpc_result.desc = 'MPC'

        mpc_result.constraints = mpcsolver.get_all_constraints()
        data_as_csv('.\data\Valid_traj_mpc.csv',t_sim,t_calc,u_sim,x_sim,y_ft_sim)
        plt.rcParams["figure.figsize"] = (20,10)

        sim_model = get_model(model_suh.model, p_sim)
        simulator = Simulator(sim_model, Ts_sim)

        x_init[:,0] = x0

        u_nn = np.nan * np.zeros((len(mpcsolver.free_inputs), n_mpc_steps))

        n_sim_steps = sim_oversampling * n_mpc_steps
        t_sim_nn = np.arange(n_sim_steps + 1) * Ts_sim
        x_sim_nn = np.nan * np.zeros((sim_model.states.numel(), n_sim_steps + 1))
        u_sim_nn = np.nan * np.zeros((sim_model.inputs.numel(), n_sim_steps + 1))
        y_sim_nn = np.nan * np.zeros((len(sim_model.y_names), n_sim_steps + 1))
        y_ft_sim_nn = np.nan * np.zeros((len(sim_model.y_names), n_sim_steps + 1))

        t_calc_nn = np.nan * np.zeros(n_sim_steps+1)

        x_sim_nn[:, 0] = x0

        for i in tqdm.tqdm(range(n_mpc_steps)):
            t_step = (np.arange(Npredict + 1) + i) * Ts_mpc

            if future_I_load_known:
                I_load_step = get_I_load(t_step)
            else:
                I_load_step = get_I_load(t_step[0])

            # NNMPC
            state_and_load = np.hstack([x_init[:,0] ,np.array(I_load_step)])
            x_nn = torch.tensor(state_and_load,dtype=torch.float32)
            t_nn_start = default_timer() 
            u_opt_nn = nnmodel(x_nn).cpu().detach() * nnmodel.label_std + nnmodel.label_mean
            t_nn_stop = default_timer() 
            u_opt_nn = u_opt_nn.numpy()
            u_opt_nn[0] = u_opt_nn[0,::-1]

            u_nn[:, i] = u_opt_nn
            

            # simulate plant
            sim_idx = sim_oversampling * i
            u_sim_cur = u_opt_nn.T
            x_sim_cur = x_sim_nn[:, sim_idx]

            sim_error = False

            for ii in range(sim_oversampling):
                try:
                    [x_sim_next, y_sim_next, y_sim_cur_ft] = simulator.sim_step(
                        x_sim_cur, u_sim_cur, feedthrough_start=True
                    )
                except RuntimeError:
                    sim_error = True
                    break

                u_sim_nn[:, [sim_idx]] = u_sim_cur
                y_ft_sim_nn[:, [sim_idx]] = y_sim_cur_ft
                t_calc_nn[sim_idx] = t_nn_stop - t_nn_start

                x_sim_nn[:, [sim_idx + 1]] = x_sim_next
                y_sim_nn[:, [sim_idx + 1]] = y_sim_next

                x_sim_cur = x_sim_next

                sim_idx = sim_idx + 1

            if sim_error:
                break

            # use plant state as "true" state, i.e. use this as the initial state
            # for the next mpc step
            x_init[:,[0]]  = x_sim_next

        nn_result = simresult.SimResult.from_data(sim_model, t_sim_nn, u_sim_nn, x_sim_nn, y_sim_nn, y_ft_sim_nn)
        nn_result.desc = 'NN Imitation Controller'

        nn_result.constraints = mpcsolver.get_all_constraints()

        plot_sim_results(
            (mpc_result,nn_result),
            plot_spec,
            reuse_figures=True,
            signal_infos=model_suh.signal_infos(),
        )
else:
        p = model_suh.params()
        model = get_model(model_suh.model, p)

        plot_spec = ("@states, lambda_O2, @inputs",)

        mpcproblem = mpc_problem.MpcProblem(model)

        mpcproblem.add_free_input("I_st", 0, np.inf, 130)
        mpcproblem.add_free_input("v_cm", 50, 250, 200)

        mpcproblem.add_constraint("p_O2", 0, np.inf)
        mpcproblem.add_constraint("p_N2", 0, np.inf)
        mpcproblem.add_constraint("p_sm", 0, np.inf)
        mpcproblem.add_constraint("w_cp", rpm2rad(20e3), rpm2rad(105e3))
        mpcproblem.add_constraint("lambda_O2", 1.5, 5, feedthrough=False)
        mpcproblem.add_constraint("lambda_O2", 1.5, 5, feedthrough=True)

        mpcproblem.add_quadratic_cost("I_st", ref="#I_load[]", weight_stage=5e-2)
        mpcproblem.add_quadratic_cost("lambda_O2", ref=2, weight_stage=1, weight_end=1)

        mpcproblem.add_quadratic_diffquot_cost(
            "v_cm", diff_quot=(-1, 1), ts_order=0, weight=1e-5
        )

        Ts_mpc = 0.025
        Npredict = 10

        future_I_load_known = False

        discretization = "rk1"
        substeps = 2
        scale_model = True
        use_lambda = True
        multiple_shooting = True

        sim_oversampling = 10
        Ts_sim = Ts_mpc / sim_oversampling
        Np = 10

        p_sim = p.copy()
        #p_sim.J_cp = p.J_cp * 0.5
        sim_model = get_model(model_suh.model, p_sim)

        simulator = Simulator(sim_model, Ts_sim)

        [u_ref, x0] = model_suh.get_validation_input(Ts_mpc)

        T_end = 29

        plugin_options = {}
        solver_options = {}

        mpcsolver = mpc_solver.MpcSolver(
            mpcproblem,
            Ts_mpc,
            Npredict,
            discretization,
            substeps=substeps,
            scale_model=scale_model,
            multiple_shooting=multiple_shooting,
            solver_options=solver_options,
            plugin_options=plugin_options,
            verbose=False,
        )

        u_init = mpcsolver.get_u_init()
        x_init = x0
        lam_g = None

        n_mpc_steps = int(np.floor(T_end / Ts_mpc))

        u_mpc = np.nan * np.zeros((len(mpcsolver.free_inputs), n_mpc_steps))
        u_fixed = np.nan * np.zeros((len(mpcsolver.fixed_inputs), n_mpc_steps))

        n_sim_steps = sim_oversampling * n_mpc_steps
        t_sim_mpc = np.arange(n_sim_steps + 1) * Ts_sim
        x_sim_mpc = np.nan * np.zeros((sim_model.states.numel(), n_sim_steps + 1))
        u_sim_mpc = np.nan * np.zeros((sim_model.inputs.numel(), n_sim_steps + 1))
        y_sim_mpc = np.nan * np.zeros((len(sim_model.y_names), n_sim_steps + 1))
        y_ft_sim_mpc = np.nan * np.zeros((len(sim_model.y_names), n_sim_steps + 1))

        x_sim_mpc[:, 0] = x0

        I_load_step = 130

        for i in tqdm.tqdm(range(n_mpc_steps)):
            t_step = (np.arange(Np + 1) + i) * Ts_mpc

            step_result = mpcsolver.solve_step(
                x0=x_init, u_init=u_init, lam_g=lam_g, I_load=I_load_step
            )

            # shift the states by one step
            # (The first state will be overwritten below by the result of the
            # simulation.)
            (x_init, u_init, lam_g) = mpcsolver.get_next_initial_values(step_result)

            u_mpc[:, i] = step_result["u_opt"][:, 0]
            u_fixed[:, i] = step_result["u_fixed"][:, 0]

            # simulate plant
            sim_idx = sim_oversampling * i
            u_sim_cur = mpcsolver.get_complete_input(
                step_result["u_opt"][:, [0]], step_result["u_fixed"][:, 0], 1
            )
            x_sim_cur = x_sim_mpc[:, sim_idx]

            sim_error = False

            for ii in range(sim_oversampling):
                try:
                    [x_sim_next, y_sim_next, y_sim_cur_ft] = simulator.sim_step(
                        x_sim_cur, u_sim_cur, feedthrough_start=True
                    )
                except RuntimeError:
                    sim_error = True
                    break

                u_sim_mpc[:, [sim_idx]] = u_sim_cur
                y_ft_sim_mpc[:, [sim_idx]] = y_sim_cur_ft

                x_sim_mpc[:, [sim_idx + 1]] = x_sim_next
                y_sim_mpc[:, [sim_idx + 1]] = y_sim_next

                x_sim_cur = x_sim_next

                sim_idx = sim_idx + 1

            if sim_error:
                break

            # use plant state as "true" state, i.e. use this as the initial state
            # for the next mpc step
            x_init[:, [0]] = x_sim_next

        mpc_result = simresult.SimResult.from_data(
            sim_model, t_sim_mpc, u_sim_mpc, x_sim_mpc, y_sim_mpc, y_ft_sim_mpc
        )

        #mpc_result.constraints = mpcsolver.get_all_constraints()

        plt.rcParams["figure.figsize"] = (20,10)
        u_nn = np.nan * np.zeros((len(mpcsolver.free_inputs), n_mpc_steps))

        n_sim_steps = sim_oversampling * n_mpc_steps
        t_sim_nn = np.arange(n_sim_steps + 1) * Ts_sim
        x_sim_nn = np.nan * np.zeros((sim_model.states.numel(), n_sim_steps + 1))
        u_sim_nn = np.nan * np.zeros((sim_model.inputs.numel(), n_sim_steps + 1))
        y_sim_nn = np.nan * np.zeros((len(sim_model.y_names), n_sim_steps + 1))
        y_ft_sim_nn = np.nan * np.zeros((len(sim_model.y_names), n_sim_steps + 1))

        x_sim_nn[:, 0] = x0

        I_load_step = 130

        for i in tqdm.tqdm(range(n_mpc_steps)):
            t_step = (np.arange(Np + 1) + i) * Ts_mpc

            #step_result = mpcsolver.solve_step(
            #    x0=x_init, u_init=u_init, lam_g=lam_g, I_load=I_load_step
            #)
            x_nn = torch.tensor(x_init[:,0],dtype=torch.float32)
            u_opt_nn = nnmodel(x_nn).cpu().detach() * nnmodel.label_std + nnmodel.label_mean
            u_opt_nn = u_opt_nn.numpy()
            u_opt_nn[0] = u_opt_nn[0,::-1]

            u_nn[:, i] = u_opt_nn

            # simulate plant
            sim_idx = sim_oversampling * i
            u_sim_cur = u_opt_nn.T
            x_sim_cur = x_sim_nn[:, sim_idx]

            sim_error = False

            for ii in range(sim_oversampling):
                try:
                    [x_sim_next, y_sim_next, y_sim_cur_ft] = simulator.sim_step(
                        x_sim_cur, u_sim_cur, feedthrough_start=True
                    )
                except RuntimeError:
                    sim_error = True
                    break

                u_sim_nn[:, [sim_idx]] = u_sim_cur
                y_ft_sim_nn[:, [sim_idx]] = y_sim_cur_ft

                x_sim_nn[:, [sim_idx + 1]] = x_sim_next
                y_sim_nn[:, [sim_idx + 1]] = y_sim_next

                x_sim_cur = x_sim_next

                sim_idx = sim_idx + 1

            if sim_error:
                break

            # use plant state as "true" state, i.e. use this as the initial state
            # for the next mpc step
            x_init[:, [0]] = x_sim_next

        nn_result = simresult.SimResult.from_data(
            sim_model, t_sim_nn, u_sim_nn, x_sim_nn, y_sim_nn, y_ft_sim_nn
        )

        #nn_result.constraints = mpcsolver.get_all_constraints()

        plot_sim_results(
            (mpc_result,nn_result),
            plot_spec,
            reuse_figures=True,
            signal_infos=model_suh.signal_infos(),
        )
# %%