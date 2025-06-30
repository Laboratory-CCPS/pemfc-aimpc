# %%
import numpy as np

from src.ctrldesign.mpc import mpc_problem, mpc_solver
from src.ctrldesign.ode_modelling.ode_model import get_model
from src.ctrldesign.ode_modelling.simulator import Simulator
from src.learning.datagen import Mpcdatagenerator
from src.model_suh import model_suh
from src.model_suh.conversions import rpm2rad

# %% Parameters

p = model_suh.params()
model = get_model(model_suh.model, p)
Ts_mpc = 1 / 50  # 0.025 # Changed to 50 Hz for Janosch
Npredict = 10
discretization_method = "rk1"
discretization_substeps = 2

future_I_load_known = False

use_lambda = True

sim_oversampling = 10
p_sim = p.copy()
p_sim.J_cp = p.J_cp * 0.5
sim_model = get_model(model_suh.model, p_sim)

Ts_sim = Ts_mpc / sim_oversampling
simulator = Simulator(sim_model, Ts_sim)
u_ref, x0 = model_suh.get_validation_input(Ts_mpc)


def get_I_load(t: np.ndarray | float) -> np.ndarray | float:
    return u_ref[1, np.minimum(np.round(t / Ts_mpc).astype(int), u_ref.shape[1])]


# %% Define options

# solver_opts = {"max_iter": 100, "dual_inf_tol": 1e-6, "compl_inf_tol": 1e-6}
solver_opts = {}
plugin_opts = {"error_on_fail": True}

# %% Setup MPC task
mpcproblem = mpc_problem.MpcProblem(model)

mpcproblem.add_free_input("I_st", 0, np.inf, 100)
mpcproblem.add_free_input("v_cm", 50, 250, 110)

mpcproblem.add_constraint("p_O2", 0, np.inf)
mpcproblem.add_constraint("p_N2", 0, np.inf)
mpcproblem.add_constraint("p_sm", 0, np.inf)
mpcproblem.add_constraint("w_cp", rpm2rad(20e3), rpm2rad(100e3))

mpcproblem.add_constraint("lambda_O2", 1.5, 5, feedthrough=False)
mpcproblem.add_constraint("lambda_O2", 1.5, 5, feedthrough=True)

mpcproblem.add_quadratic_cost("I_st", ref="#I_load[]", weight_stage=5e-2)
mpcproblem.add_quadratic_cost("lambda_O2", ref=2, weight_stage=1, weight_end=1)

mpcproblem.add_quadratic_diffquot_cost(
    "v_cm", diff_quot=(-1, 1), ts_order=0, weight=1e-5, prev_value="#v_cm_prev"
)

# %% Instantiate

mpcsolver = mpc_solver.MpcSolver(
    mpcproblem,
    Ts_mpc,
    Npredict,
    discretization_method,
    substeps=discretization_substeps,
    scale_model=True,
    use_lambda=use_lambda,
    multiple_shooting=True,
    verbose=False,
    solver_options=solver_opts,
    plugin_options=plugin_opts,
    use_opti_function=True,
)

data_gen = Mpcdatagenerator(
    mpcsolver, n_samples=20000, load_high=350, load_low=80, filename="50Hz"
)

# %% Generate Data
file = data_gen.generate_data()
