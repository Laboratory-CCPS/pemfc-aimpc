# %%
import os

import tqdm

# disable oneDNN opts
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from timeit import default_timer

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import rc

from src.ctrldesign.mpc import mpc_problem, mpc_solver
from src.ctrldesign.ode_modelling import simresult
from src.ctrldesign.ode_modelling.ode_model import get_model
from src.ctrldesign.ode_modelling.simresult import AddSignal, plot_sim_results
from src.ctrldesign.ode_modelling.simulator import Simulator
from src.learning.utils import data_as_csv, plot_dataframes
from src.model_suh import model_suh
from src.model_suh.conversions import rpm2rad

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


def get_validation_input_60(Ts: float) -> tuple[np.ndarray, np.ndarray]:
    t0 = 0
    tend = 60

    t = np.arange(t0, tend, Ts).reshape((1, -1))

    def I_load(t: np.ndarray) -> np.ndarray:
        return (
            100.0
            + 80 * (t >= 9)
            + 40 * (t >= 19)
            - 20 * (t >= 29)
            + 60 * (t >= 39)
            + 60 * (t >= 55)
        )

    def v_cm(t: np.ndarray) -> np.ndarray:
        return (
            100.0
            + 55 * (t >= 9)
            + 25 * (t >= 19)
            - 10 * (t >= 29)
            + 40 * (t >= 39)
            + 25 * (t >= 49)
            + 15 * (t >= 59)
        )

    x0 = np.array([0.1096e5, 0.7502e5, 5.4982e3, 1.4326e5])

    u = np.vstack((v_cm(t), I_load(t)))

    return (u, x0)


[u_ref, x0] = get_validation_input_60(Ts_mpc)


def get_I_load(t: np.ndarray | float) -> np.ndarray | float:
    return u_ref[1, np.minimum(np.round(t / Ts_mpc).astype(int), u_ref.shape[1])]


T_end = 60

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

# %% Save Valid MPC

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

plot_sim_results(
    (mpc_result),
    plot_spec,
    add_signals={
        "I_load": AddSignal(
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

save_until = len(t_sim) - 1

data_as_csv(
    f".\data\{T_end}s_valid_traj_mpc_50Hz.csv",
    np.round(t_sim[:save_until], 2),
    t_calc[:save_until],
    u_sim[:, :save_until],
    x_sim[:, :save_until],
    y_ft_sim[:, :save_until],
    I_load=I_load.flatten(),
    as_rpm=False,
)

# %%
nnmodel = tf.keras.models.load_model("./data/nn_mpc_model.keras")
mpc_data_df = pd.read_csv(r".\data\60s_valid_traj_mpc_50Hz.csv")

fnames = [
    "p_O2",  # pressure O2
    "p_N2",  # pressure N2
    "w_cp",  # rotations compressor
    "p_sm",  # pessure after compressor
    "I_load",  # load profile current
]


lnames = ["I_st", "v_cm"]  # stack current and compressor voltage

u_opt_nn = np.nan * np.zeros((n_mpc_steps, len(lnames)))
t_calc_nn = np.nan * np.zeros(n_sim_steps)

for i, row in tqdm.tqdm(mpc_data_df.iterrows()):
    t_nn_start = default_timer()
    u_opt_nn[i, :] = nnmodel(tf.constant(row[fnames].values.reshape(-1, len(fnames))))
    t_nn_stop = default_timer()
    t_calc_nn[i] = t_nn_stop - t_nn_start


nn_data_df = mpc_data_df.copy(deep=True)
nn_data_df[lnames] = u_opt_nn
nn_data_df["t_calc"] = t_calc_nn
# %% Comparison Plot
# Enable LaTeX font rendering

rc("text", usetex=True)
rc("font", family="serif")

mpc_data_df = mpc_data_df.drop(columns=["lambda_O2"])
mpc_data_df.to_csv(r".\data\60s_valid_traj_mpc_50Hz.csv")

nn_data_df = nn_data_df.drop(columns=["lambda_O2"])
nn_data_df.to_csv(r".\data\60s_valid_traj_nnmpc_50Hz.csv")

plot_dataframes(
    [mpc_data_df, nn_data_df],
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
    },
    const_high={"v_cm": 250, "w_cp": 100},
    names=["Baseline MPC", "NN Imitation Controller"],
    figsize=(28, 14),
    is_rpm=False,
    do_save=True,
)
# %%
