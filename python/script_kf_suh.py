# %%
import numpy as np
from matplotlib import pyplot as plt

from src.ctrldesign.estimation.ekf import ExtendedKalmanFilter
from src.ctrldesign.ode_modelling.model_helper import (
    scale_model_signals,
)
from src.ctrldesign.ode_modelling.ode_model import (
    get_linearized_matrices,
    get_model,
    scale_model,
    set_outputs,
)
from src.ctrldesign.ode_modelling.simresult import SimResult, plot_sim_results
from src.ctrldesign.ode_modelling.simulator import sim_nl
from src.model_suh import model_suh
from src.model_suh.conversions import rpm2rad

#! %matplotlib qt

# %% Load model with standard parameters

# measured output signals
meas_outputs = ["w_cp", "p_sm", "v_st"]

# parameters of true system
p = model_suh.params()

# parameters assumed by kalman filter
p_kf = model_suh.params()
p_kf.eta_cp *= 0.9
# p_kf.V_sm *= 1.05
# p_kf.T_st = celsius2kelvin(70)

# model is used to simulate the "true" data, kfmodel is the model which is used within
# the kalman filter
model = set_outputs(get_model(model_suh.model, p), meas_outputs)
kfmodel = set_outputs(get_model(model_suh.model, p_kf), meas_outputs)


# %% Simulate continuous model

Ts = 0.025
(u, x0) = model_suh.get_validation_input(Ts)

t_max = 25
idx_max = int(t_max // Ts)
u = u[:, :idx_max]

res = sim_nl(model, Ts, x0, u)
res.desc = f"cont., Ts = {Ts} s"

plot_sim_results(res, signal_infos=model_suh.signal_infos())


# %% Analysis of observability


def get_obsv_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    q, n = C.shape
    m = np.zeros((q * n, n))

    m[:q, :] = C

    Ap = np.eye(n)

    for i in range(1, q):
        Ap = Ap @ A
        m[i * q : (i + 1) * q] = C @ Ap

    return m


u_pre_change_idx = set(np.where(u[0, 1:] != u[0, :-1])[0])
u_pre_change_idx.union(set(np.where(u[1, 1:] != u[1, :-1])[0].flatten()))

u_pre_change_idx = sorted(u_pre_change_idx)

scaled = True

fmodel = kfmodel

if scaled:
    fmodel = scale_model(fmodel)

for idx in u_pre_change_idx:

    if scaled:
        xk = scale_model_signals(fmodel, "x", res.x[:, idx])
        uk = scale_model_signals(fmodel, "u", res.u[:, idx])
    else:
        xk = res.x[:, idx]
        uk = res.u[:, idx]

    (A, B, C, D) = get_linearized_matrices(fmodel, xk, uk)

    obs = get_obsv_matrix(A, C)
    print(np.linalg.svdvals(obs), np.linalg.cond(obs))


# %% Kalman filter

ekf = ExtendedKalmanFilter(
    model=kfmodel,
    Ts=Ts,
    discretization="rk1",
    substeps=4,
    P_discretization="rk1",
    P_substeps=4,
    scale_model=True,
    Q=np.array([0.075**2, 0.075**2, 0.075**2, 0.075**2]),
    scaled_Q=True,
    R=np.array([0.01**2, 0.05**2, 0.02**2]),
    scaled_R=True,
)

np.random.seed(42)
y_meas = res.get_signal_values(meas_outputs)
N = y_meas.shape[1]
y_meas = y_meas + np.vstack(
    (
        np.random.normal(0, rpm2rad(105 * 1000) * 0.01, (1, N)),
        np.random.normal(0, 2e5 * 0.1, (1, N)),
        np.random.normal(0, 40 * 0.02, (1, N)),
    )
)
y_meas[1, :] = y_meas[1, :].clip(0.025e5, 4.5e5)

x0_noise = x0 * np.array([2, 0.25, 1.02, 1.2])

ekf.init(
    x0_noise,
    P0=np.array([1, 1, 0.05, 0.2]),
    P0_scaled=True,
)

x_est = np.nan * np.ones_like(res.x)

for i in range(0, len(res.t) - 1):
    x_est[:, i] = ekf.get_estimate()

    if i >= 0:
        ekf.do_meas(y_meas[:, i], res.u[:, i])

    ekf.do_step(res.u[:, i])

x_est[:, -1] = ekf.get_estimate()


kfres = SimResult.from_data(kfmodel, res.t.copy(), res.u.copy(), x_est, y_meas)
kfres.desc = "kf est"

plot_sim_results((res, kfres), signal_infos=model_suh.signal_infos())

# %%
plt.subplot(3, 1, 1)
plt.plot(kfres.t, y_meas[1, :].T)
plt.plot(kfres.t, kfres.get_signal_values("p_sm").T)
plt.plot(kfres.t, res.get_signal_values("p_sm").T)
