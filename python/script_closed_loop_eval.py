# %% Imports
from timeit import default_timer

import numpy as np
import pandas as pd
import torch
import tqdm
from matplotlib import rc

from src.ctrldesign.ode_modelling.ode_model import get_model
from src.ctrldesign.ode_modelling.simulator import Simulator
from src.learning.utils import data_as_csv, load_scaled_constraint_NN, plot_dataframes
from src.model_suh import model_suh

# Enable LaTeX font rendering
rc("text", usetex=True)
rc("font", family="serif")


# %% Loading
NN_path = "./current_best_nn.pickle"

nnmodel = load_scaled_constraint_NN(NN_path)
# %% Closed-loop evaluation
p = model_suh.params()
model = get_model(model_suh.model, p)

plot_spec = ("@states, lambda_O2, @inputs",)

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
# p_sim.J_cp = p.J_cp * 0.8
sim_model = get_model(model_suh.model, p_sim)

simulator = Simulator(sim_model, Ts_sim)

[u_ref, x0] = model_suh.get_validation_input(Ts_mpc)


def get_I_load(t: np.ndarray | float) -> np.ndarray | float:
    return u_ref[1, np.minimum(np.round(t / Ts_mpc).astype(int), u_ref.shape[1])]


T_end = 29

x_init = x0
lam_g = None

n_mpc_steps = int(np.floor(T_end / Ts_mpc))

sim_model = get_model(model_suh.model, p_sim)
simulator = Simulator(sim_model, Ts_sim)

x_init = x0.reshape(-1, 1)

u_nn = np.nan * np.zeros((2, n_mpc_steps))

n_sim_steps = sim_oversampling * n_mpc_steps
t_sim_nn = np.arange(n_sim_steps + 1) * Ts_sim
x_sim_nn = np.nan * np.zeros((sim_model.states.numel(), n_sim_steps + 1))
u_sim_nn = np.nan * np.zeros((sim_model.inputs.numel(), n_sim_steps + 1))
y_sim_nn = np.nan * np.zeros((len(sim_model.y_names), n_sim_steps + 1))
y_ft_sim_nn = np.nan * np.zeros((len(sim_model.y_names), n_sim_steps + 1))

t_calc_nn = np.nan * np.zeros(n_sim_steps + 1)

x_sim_nn[:, 0] = x0
x_init = x0.reshape(-1, 1)

for i in tqdm.tqdm(range(n_mpc_steps)):
    t_step = (np.arange(Np + 1) + i) * Ts_mpc

    if future_I_load_known:
        I_load_step = get_I_load(t_step)
    else:
        I_load_step = get_I_load(t_step[0])

    # NNMPC
    state_and_load = np.hstack([x_init[:, 0], np.array(I_load_step)])
    x_nn = torch.tensor(state_and_load, dtype=torch.float32)
    t_nn_start = default_timer()
    u_opt_nn = nnmodel(x_nn).cpu().detach() * nnmodel.label_std + nnmodel.label_mean
    t_nn_stop = default_timer()
    u_opt_nn = u_opt_nn.numpy()
    u_opt_nn[0] = u_opt_nn[0, ::-1]

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
    x_init[:, [0]] = x_sim_next

nn_df = data_as_csv(
    "./data/Valid_traj_nn.csv",
    t_sim_nn,
    t_calc_nn,
    u_sim_nn,
    x_sim_nn,
    y_ft_sim_nn,
    output_df=True,
)

# %% Plotting for evaluation
mpc_df = pd.read_csv("./data/Valid_traj_mpc.csv")
gp_df = pd.read_csv("./data/Valid_traj_gp_good.csv")


def subsample_dataframe(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Subsamples a pandas DataFrame to include every nth point per column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to be subsampled.
    - n (int): The interval at which to sample (e.g., every nth row).

    Returns:
    - pd.DataFrame: A DataFrame containing only the sampled points.
    """
    if n < 1:
        raise ValueError("n must be a positive integer.")

    return df.iloc[::n]


subsampled_mpc_df = subsample_dataframe(mpc_df, 10)
subsampled_nn_df = subsample_dataframe(nn_df, 10)
subsampled_gp_df = subsample_dataframe(gp_df, 10)


plot_dataframes(
    [subsampled_mpc_df, subsampled_nn_df, subsampled_gp_df],
    "t_sim",
    ["t_calc"],
    ["p_O2", "p_N2", "p_sm", "w_cp"],
    const_low={
        "I_st": 0,
        "v_cm": 50,
        "p_O2": 0,
        "p_N2": 0,
        "p_sm": 0,
        "w_cp": 20,
        "lambda_O2": 1.5,
    },
    const_high={"v_cm": 250, "w_cp": 105},  # ,'lambda_O2':5},
    names=["Baseline MPC", "NN Imitation Controller", "GP Imitation Controller"],
    figsize=(28, 14),
    do_save=True,
)

# %%
