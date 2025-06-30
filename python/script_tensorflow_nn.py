# %%
import os

import tqdm

# disable oneDNN opts
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.ctrldesign.mpc import mpc_problem, mpc_solver
from src.ctrldesign.ode_modelling import simresult
from src.ctrldesign.ode_modelling.ode_model import get_model
from src.ctrldesign.ode_modelling.simresult import AddSignal, plot_sim_results
from src.ctrldesign.ode_modelling.simulator import Simulator
from src.learning.utils import data_as_csv
from src.model_suh import model_suh
from src.model_suh.conversions import rpm2rad

# %% Data

full_df = pd.read_csv("./data/50Hz_MPC_10_samples_20000_datapoints_2025_01_30_11.csv")

fnames = [
    "p_O2",  # pressure O2
    "p_N2",  # pressure N2
    "w_cp",  # rotations compressor
    "p_sm",  # pessure after compressor
    "I_load",
]  # load profile current


lnames = ["I_st", "v_cm"]  # stack current  # compressor voltage


X = full_df[fnames].values
y = full_df[lnames].values

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %% Scaling

# Features
feature_scaler = StandardScaler()
feature_scaler.fit(X_train)
feature_mean = feature_scaler.mean_
feature_std = feature_scaler.scale_

# Labels
label_scaler = StandardScaler()
label_scaler.fit(y_train)
label_mean = label_scaler.mean_
label_std = label_scaler.scale_

# Output constraints
const_high = tf.constant([500, 250], dtype=tf.float32)
const_low = tf.constant([0, 50], dtype=tf.float32)

const_high_scaled = (const_high - label_mean) / label_std
const_low_scaled = (const_low - label_mean) / label_std

const_scale_scaled = const_high_scaled - const_low_scaled

# In-/Output dimensions
n_inputs = X_train.shape[1]
n_outputs = y_train.shape[1]

# %% Define the Sequential model

# Given parameters
n_neurons = 10
n_layers = 2
activation = "sigmoid"

nnmodel = tf.keras.Sequential(
    [
        # Input
        tf.keras.layers.Input(shape=(n_inputs,)),
        # Feature normalization
        tf.keras.layers.Normalization(
            axis=-1, mean=feature_mean, variance=feature_std**2
        ),
        # Hidden layers
        tf.keras.layers.Dense(
            n_neurons,
            activation=activation,
            kernel_regularizer=tf.keras.regularizers.L2(0.25),
        ),
        tf.keras.layers.Dense(
            n_neurons,
            activation=activation,
            kernel_regularizer=tf.keras.regularizers.L2(0.25),
        ),
        # Output layer with 'sigmoid' activation (constrains values between 0 and 1)
        tf.keras.layers.Dense(
            n_outputs,
            activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.L2(0.25),
        ),
        # First rescaling: map sigmoid output to [output_low_scaled, output_high_scaled]
        tf.keras.layers.Rescaling(scale=const_scale_scaled, offset=const_low_scaled),
        # Final rescaling: adjust to the original label scale
        tf.keras.layers.Rescaling(scale=label_std, offset=label_mean),
    ]
)

# Model summary
print(nnmodel.summary())

# %% Training
num_epochs = 1500
batch_size = 512

nnmodel.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0006),
    loss="mse",
    metrics=[tf.keras.metrics.R2Score()],
)

hist = nnmodel.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=num_epochs,
    batch_size=batch_size,
    verbose=1,
)


# Plotting
fig, ax = plt.subplots(2, 1)
ax[0].plot(range(num_epochs), hist.history["loss"])
ax[0].plot(range(num_epochs), hist.history["val_loss"])
ax[0].legend(["train loss", "validation loss"])
ax[0].set_yscale("log")

ax[1].plot(range(num_epochs), hist.history["val_r2_score"])

# %% Closed-loop test

p = model_suh.params()
model = get_model(model_suh.model, p)

plot_spec = ("@states, lambda_O2, @inputs",)

mpcproblem = mpc_problem.MpcProblem(model)

mpcproblem.add_free_input("I_st", 0, np.inf, 100)
mpcproblem.add_free_input("v_cm", 50, 250, 110)

mpcproblem.add_constraint("p_O2", 0, np.inf)
mpcproblem.add_constraint("p_N2", 0, np.inf)
mpcproblem.add_constraint("p_sm", 0, np.inf)
mpcproblem.add_constraint("w_cp", rpm2rad(20e3), rpm2rad(100e3))
mpcproblem.add_constraint("lambda_O2", 1.5, 5, feedthrough=True)
mpcproblem.add_constraint("lambda_O2", 1.5, 5, feedthrough=False)

mpcproblem.add_quadratic_cost("I_st", ref="#I_load", weight_stage=5e-2)
mpcproblem.add_quadratic_cost("lambda_O2", ref=2, weight_stage=1, weight_end=1)

mpcproblem.add_quadratic_diffquot_cost(
    "v_cm", diff_quot=(-1, 1), ts_order=0, weight=1e-5
)

Ts_mpc = 0.02
Npredict = 10

future_I_load_known = False

discretization = "rk1"
substeps = 2
scale_model = True
use_lambda = True
multiple_shooting = True

sim_oversampling = 1
Ts_sim = Ts_mpc / sim_oversampling
p_sim = p.copy()
# p_sim.J_cp = p.J_cp * 0.5
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

t_load = np.nan * np.zeros((n_mpc_steps,))
I_load = np.nan * np.zeros((1, n_mpc_steps))

n_sim_steps = sim_oversampling * n_mpc_steps
t_sim = np.arange(n_sim_steps + 1) * Ts_sim
x_sim = np.nan * np.zeros((sim_model.states.numel(), n_sim_steps + 1))
u_sim = np.nan * np.zeros((sim_model.inputs.numel(), n_sim_steps + 1))
y_sim = np.nan * np.zeros((len(sim_model.y_names), n_sim_steps + 1))
y_ft_sim = np.nan * np.zeros((len(sim_model.y_names), n_sim_steps + 1))

t_calc = np.nan * np.zeros(n_sim_steps + 1)

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

    if isinstance(I_load_step, np.ndarray):
        I_load[:, i] = I_load_step[0]
        t_load[i] = t_step[0]
    else:
        I_load[:, i] = I_load_step
        t_load[i] = t_step[0]

    # simulate plant
    sim_idx = sim_oversampling * i
    u_sim_cur = mpcsolver.construct_model_input(
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
mpc_result.desc = "MPC"

mpc_result.constraints = mpcsolver.get_all_constraints()
data_as_csv(
    ".\data\Keras_valid_traj_mpc_50Hz.csv", t_sim, t_calc, u_sim, x_sim, y_ft_sim
)
plt.rcParams["figure.figsize"] = (20, 10)

sim_model = get_model(model_suh.model, p_sim)
simulator = Simulator(sim_model, Ts_sim)

x_init[:, 0] = x0

u_nn = np.nan * np.zeros((len(mpcsolver.free_inputs), n_mpc_steps))
t_load = np.nan * np.zeros((n_mpc_steps,))
I_load = np.nan * np.zeros((1, n_mpc_steps))

n_sim_steps = sim_oversampling * n_mpc_steps
t_sim_nn = np.arange(n_sim_steps + 1) * Ts_sim
x_sim_nn = np.nan * np.zeros((sim_model.states.numel(), n_sim_steps + 1))
u_sim_nn = np.nan * np.zeros((sim_model.inputs.numel(), n_sim_steps + 1))
y_sim_nn = np.nan * np.zeros((len(sim_model.y_names), n_sim_steps + 1))
y_ft_sim_nn = np.nan * np.zeros((len(sim_model.y_names), n_sim_steps + 1))

t_calc_nn = np.nan * np.zeros(n_sim_steps + 1)

x_sim_nn[:, 0] = x0

for i in tqdm.tqdm(range(n_mpc_steps)):
    t_step = (np.arange(Npredict + 1) + i) * Ts_mpc

    if future_I_load_known:
        I_load_step = get_I_load(t_step)
    else:
        I_load_step = get_I_load(t_step[0])

    # NNMPC
    state_and_load = np.hstack([x_init[:, 0], np.array(I_load_step)]).reshape(
        1, n_inputs
    )

    x_nn = tf.constant(state_and_load, dtype=tf.float32)
    t_nn_start = default_timer()
    u_opt_nn = nnmodel(x_nn)
    t_nn_stop = default_timer()
    u_opt_nn = u_opt_nn.numpy()
    u_opt_nn[0] = u_opt_nn[0, ::-1]

    u_nn[:, i] = u_opt_nn

    if isinstance(I_load_step, np.ndarray):
        I_load[:, i] = I_load_step[0]
        t_load[i] = t_step[0]
    else:
        I_load[:, i] = I_load_step
        t_load[i] = t_step[0]

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

nn_result = simresult.SimResult.from_data(
    sim_model, t_sim_nn, u_sim_nn, x_sim_nn, y_sim_nn, y_ft_sim_nn
)
nn_result.desc = "NN Imitation Controller"

nn_result.constraints = mpcsolver.get_all_constraints()

plot_sim_results(
    (mpc_result, nn_result),
    plot_spec,
    add_signals={
        "I_st": AddSignal(
            I_load,
            t=t_load,
            label="I_load",
            on_top=False,
            step_plot=True,
            format_kwargs={"linewidth": 2, "linestyle": "--", "color": "gray"},
        )
    },
    reuse_figures=True,
    signal_infos=model_suh.signal_infos(),
)
# %%
nnmodel.save("./data/nn_mpc_model.keras")

loaded_nnmodel = tf.keras.models.load_model("./data/nn_mpc_model.keras")

print(f"Before save: {nnmodel(tf.constant(X_train[0,:].reshape(-1,5)))}")
print(f"After save: {loaded_nnmodel(tf.constant(X_train[0,:].reshape(-1,5)))}")

# %%
