# %%
# import importlib

import numpy as np
import tqdm

from src.ctrldesign.mpc import mpc_problem, mpc_solver
from src.ctrldesign.ode_modelling import simresult
from src.ctrldesign.ode_modelling.discrete_model import discretize_model
from src.ctrldesign.ode_modelling.ode_model import get_model
from src.ctrldesign.ode_modelling.simresult import AddSignal, plot_sim_results
from src.ctrldesign.ode_modelling.simulator import Simulator, sim_nl
from src.model_suh import model_suh
from src.model_suh.conversions import rpm2rad

#! %matplotlib qt

# %% Load model with standard parameters

p = model_suh.params()
model = get_model(model_suh.model, p)

plot_spec = ("@states, lambda_O2, @inputs",)


# %% Setup MPC problem
# (used in all exampled that follow)

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


# %% Parameters for MPC solver
# (used in all exampled that follow)

Ts_mpc = 0.025
Npredict = 10

discretization = "rk1"
substeps = 2
scale_model = True
use_lambda = True
multiple_shooting = True

# options forwared to casadi
plugin_options = {}
solver_options = {}
# solver_options = {"max_iter": 100, "dual_inf_tol": 1e-6, "compl_inf_tol": 1e-6}


# %% Example 1: Solve the optimization problem for one step

# x = [p_O2, p_N2, w_CP, p_sm]
x0 = np.array([[0.1113e5, 0.8350e5, 6049.9, 1.5356e5]]).T


def get_I_load(t: np.ndarray | float) -> np.ndarray | float:
    return 130 + (t >= 0.15) * 100


mpcsolver = mpc_solver.MpcSolver(
    mpcproblem,
    Ts_mpc,
    Npredict,
    discretization,
    substeps=substeps,
    scale_model=scale_model,
    use_lambda=use_lambda,
    multiple_shooting=multiple_shooting,
    solver_options=solver_options,
    plugin_options=plugin_options,
    verbose=False,
)

# for parameters that are defined as vectors in the mpc problem setup the
# provided value must be either a scalar (which is then used for every time
# point) or as a vector of length N + 1, corresponding to the time points
# k = 0, ..., N, even if not all time points are used.
I_load = get_I_load(np.arange(0, Npredict + 1) * Ts_mpc)

u_init = mpcsolver.get_u_init()

step_result = mpcsolver.solve_step(x0=x0, u_init=u_init, lam_g=None, I_load=I_load)

mpc_result = mpcsolver.sol_into_simresult(step_result)

plot_sim_results(
    mpc_result,
    plot_spec,
    add_signals={
        "I_st": AddSignal(
            I_load,
            label="I_load",
            on_top=False,
            step_plot=True,
            format_kwargs={"linewidth": 2, "linestyle": "--", "color": "gray"},
        )
    },
    reuse_figures=True,
    signal_infos=model_suh.signal_infos(),
)


# %% Example 2: Simulate the MPC with ideal model
# (That is, use the solution x(k + 1) of the optimization problem as
# state of the next step.)

# If future_I_load_known is true, the MPC is provided with the demanded
# current of each of the N time points.
future_I_load_known = False

sim_oversampling = 1
Ts_sim = Ts_mpc / sim_oversampling

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
    use_lambda=use_lambda,
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
I_load = np.nan * np.zeros((1, n_mpc_steps + 1))

for i in tqdm.tqdm(range(n_mpc_steps)):

    if future_I_load_known:
        t_step = (np.arange(mpcsolver.N) + i) * mpcsolver.Ts
        I_load_step = get_I_load(t_step)
    else:
        t_step = i * mpcsolver.Ts
        I_load_step = get_I_load(t_step)

    step_result = mpcsolver.solve_step(
        x0=x_init, u_init=u_init, lam_g=lam_g, I_load=I_load_step
    )

    (x_init, u_init, lam_g) = mpcsolver.get_next_initial_values(step_result)

    u_mpc[:, i] = step_result["u_opt"][:, 0]
    u_fixed[:, i] = step_result["u_fixed"][:, 0]

    if isinstance(I_load_step, np.ndarray):
        I_load[:, i] = I_load_step[0]
    else:
        I_load[:, i] = I_load_step

u_sim = mpcsolver.construct_model_input(u_mpc, u_fixed, sim_oversampling)
idealmpc_result = sim_nl(model, Ts_sim, x0, u_sim)
idealmpc_result.constraints = mpcsolver.get_all_constraints()
idealmpc_result.desc = "mpc with ideal model"

plot_sim_results(
    idealmpc_result,
    plot_spec,
    add_signals={
        "I_st": AddSignal(
            I_load,
            label="I_load",
            on_top=False,
            step_plot=True,
            format_kwargs={"linewidth": 2, "linestyle": "--", "color": "gray"},
        )
    },
    reuse_figures=True,
    signal_infos=model_suh.signal_infos(),
)


# %% Example 3: Simulate the MPC with seperately simulated model

# If future_I_load_known is true, the MPC is provided with the demanded
# current of each of the N time points.
future_I_load_known = False

sim_oversampling = 1
Ts_sim = Ts_mpc / sim_oversampling

Np = 10

# Define true plant for simulation
# If the MPC ist implemented using 'discretization' and 'substeps', the following
# should be the simulation model to have a perfect match:
#
# sim_oversampling = substeps
# Ts_sim = Ts_mpc / sim_oversampling
# sim_model = discretize_model(model, Ts_sim, discretization)


# 10 times oversampling (i.e. 10 simulation steps to simulate one mpc step)
# same modelo, but using the "idas" solver
#
# sim_oversampling = 10
# Ts_sim = Ts_mpc / sim_oversampling
# sim_model = model

# 10 times oversampling, "idas" solver, parameter mismatch
#
sim_oversampling = 10
Ts_sim = Ts_mpc / sim_oversampling
p_sim = p.copy()
p_sim.J_cp = p.J_cp * 0.5
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
    use_lambda=use_lambda,
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
t_load = np.nan * np.zeros((n_mpc_steps,))
I_load = np.nan * np.zeros((1, n_mpc_steps))

n_sim_steps = sim_oversampling * n_mpc_steps
t_sim = np.arange(n_sim_steps + 1) * Ts_sim
x_sim = np.nan * np.zeros((sim_model.states.numel(), n_sim_steps + 1))
u_sim = np.nan * np.zeros((sim_model.inputs.numel(), n_sim_steps + 1))
y_sim = np.nan * np.zeros((len(sim_model.y_names), n_sim_steps + 1))
y_ft_sim = np.nan * np.zeros((len(sim_model.y_names), n_sim_steps + 1))

x_sim[:, 0] = x0

for i in tqdm.tqdm(range(n_mpc_steps)):
    t_step = (np.arange(Np + 1) + i) * Ts_mpc

    if future_I_load_known:
        I_load_step = get_I_load(t_step)
    else:
        I_load_step = get_I_load(t_step[0])

    step_result = mpcsolver.solve_step(
        x0=x_init, u_init=u_init, lam_g=lam_g, I_load=I_load_step
    )

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

# plot_sim_results(mpc_result, plot_spec, reuse_figures=True, signal_infos=model_suh.signal_infos())
plot_sim_results(
    (idealmpc_result, mpc_result),
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
