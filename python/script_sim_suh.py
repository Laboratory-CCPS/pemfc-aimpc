# %%
import numpy as np

from src.ctrldesign.ode_modelling.discrete_model import discretize_model
from src.ctrldesign.ode_modelling.ode_model import get_linearized_matrices, get_model
from src.ctrldesign.ode_modelling.simresult import plot_sim_results
from src.ctrldesign.ode_modelling.simulator import sim_nl
from src.model_suh import model_suh

#! %matplotlib qt

# %% Load model with standard parameters

p = model_suh.params()
model = get_model(model_suh.model, p)


# %% Simulate continuous model
Ts = 0.01
(u, x0) = model_suh.get_validation_input(Ts)
u = u[:, : int(25 // Ts)]

res = sim_nl(model, Ts, x0, u)
res.desc = f"cont., Ts = {Ts} s"

plot_sim_results(res, signal_infos=model_suh.signal_infos())


# %% Simulate discretized model

Ts = 0.001
dmodel = discretize_model(model, Ts, "rk4")

(u, x0) = model_suh.get_validation_input(Ts)
u = u[:, : int(25 // Ts)]

dres = sim_nl(dmodel, Ts, x0, u)
dres.desc = f"disc., Ts = {Ts} s"

plot_sim_results((dres, res), signal_infos=model_suh.signal_infos())


# %% Have a look at the eigenvalues of the linearized system

u_pre_change_idx = set(np.where(res.u[0, 1:] != res.u[0, :-1])[0])
u_pre_change_idx.union(set(np.where(res.u[1, 1:] != res.u[1, :-1])[0].flatten()))

u_pre_change_idx = sorted(u_pre_change_idx)

scaled = True

for idx in u_pre_change_idx:

    tk = res.t[idx]
    xk = res.x[:, idx]
    uk = res.u[:, idx]

    (A, _, _, _) = get_linearized_matrices(model, xk, uk)

    evs = np.linalg.eigvals(A)
    evs.sort()

    print(f"eigenvalues of A evaluated at (x({tk:.2f}), u({tk:.2f})):", evs)
