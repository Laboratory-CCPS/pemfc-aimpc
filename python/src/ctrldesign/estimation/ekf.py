from collections.abc import Sequence
from typing import Literal, Optional

import casadi
import numpy as np

from ..ode_modelling import ode_model
from ..ode_modelling.casadi_helper import casadi_vars_to_str
from ..ode_modelling.discrete_model import discretize_model
from ..ode_modelling.model_helper import (
    scale_model_signals,
    unscale_model_signals,
)
from ..ode_modelling.ode_model import OdeModel, get_linearized_matrices
from ..ode_modelling.scaling import Scaling
from ..ode_modelling.simulator import Simulator

DiscretizatonType = Literal["rk1", "rk4", "idas"]


def _scale_covariance_matrix(
    scalings: dict[str, Scaling], names: Sequence[str], data: np.ndarray
) -> np.ndarray:
    factors = np.array([scalings[name].scale_derivate(1.0) for name in names]).reshape(
        (-1, 1)
    )
    factors = factors @ factors.T

    return data * factors


def _unscale_covariance_matrix(
    scalings: dict[str, Scaling], names: Sequence[str], data: np.ndarray
) -> np.ndarray:
    factors = np.array(
        [scalings[name].unscale_derivate(1.0) for name in names]
    ).reshape((-1, 1))
    factors = factors @ factors.T

    return data * factors


def _sim_covariance(
    P: np.ndarray,
    A: np.ndarray,
    Q: np.ndarray,
    Ts: float,
    method: Literal["rk1", "rk4"],
    substeps: int,
):
    dt = Ts / substeps

    def ode(P: np.ndarray) -> np.ndarray:
        return A @ P + P @ A.T + Q

    if method == "rk1":
        for _ in range(substeps):
            P = P + dt * ode(P)
    else:
        for _ in range(substeps):
            k1 = ode(P)
            k2 = ode(P + dt / 2 * k1)
            k3 = ode(P + dt / 2 * k2)
            k4 = ode(P + dt * k3)
            P = P + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return 0.5 * (P + P.T)


class ExtendedKalmanFilter:
    def __init__(
        self,
        model: OdeModel,
        Ts: float,
        discretization: DiscretizatonType,
        P_discretization: DiscretizatonType,
        substeps: int = 1,
        P_substeps: int = 1,
        scale_model: bool = False,
        Q: Optional[np.ndarray] = None,
        scaled_Q: bool = False,
        R: Optional[np.ndarray] = None,
        scaled_R: bool = False,
    ):
        assert P_discretization in ("rk1", "rk4")

        self.scaled = scale_model
        self.orig_scaled = model.scaled

        self.scalings = dict[str, Scaling]

        if self.scaled:
            self.model = ode_model.scale_model(model)
            self.scalings = self.model.scalings.copy()
        else:
            self.model = model
            self.scalings = {}

        self.x_names = casadi_vars_to_str(self.model.states)

        self.Ts = Ts
        self.discretization = discretization
        self.substeps = substeps
        self.P_discretization = P_discretization
        self.P_substeps = P_substeps

        if discretization == "idas":
            self.simulator = Simulator(self.model, Ts)
        else:
            self.simulator = Simulator(
                discretize_model(self.model, Ts, discretization, substeps), Ts
            )

        self.f_out = casadi.Function(
            "outputs", [self.model.states, self.model.inputs], [self.model.y]
        )

        if len(Q.shape) == 1:
            Q = np.diag(Q)

        if len(R.shape) == 1:
            R = np.diag(R)

        if self.scaled:
            if not scaled_Q:
                Q = _scale_covariance_matrix(self.model.scalings, self.x_names, Q)

            if not scaled_R:
                R = _scale_covariance_matrix(self.model.scalings, self.model.y_names, R)

        self.Q = Q
        self.R = R

        (A, _, C, _) = get_linearized_matrices(self.model)
        self.A: casadi.Function = A
        self.C: casadi.Function = C

    def init(self, x0: np.ndarray, P0: np.ndarray, P0_scaled: bool = False):
        if self.scaled:
            x0 = scale_model_signals(self.model, "x", x0)

        if len(P0.shape) == 1:
            P0 = np.diag(P0)

        if not P0_scaled:
            P0 = _scale_covariance_matrix(self.model.scalings, self.x_names, P0)
        else:
            P0 = P0.copy()

        self.x = x0
        self.P = P0

    def do_step(self, u: np.ndarray, Q: Optional[np.ndarray] = None):
        assert Q is None, "not yet implemented"

        if self.scaled:
            u = scale_model_signals(self.model, "u", u)

        A = self.A(self.x, u).full()

        self.P = _sim_covariance(
            self.P, A, self.Q, self.Ts, self.P_discretization, self.P_substeps
        )

        self.x = self.simulator.sim_step(self.x, u)[0].reshape((-1,))

    def do_meas(
        self,
        y_meas: np.ndarray,
        u: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ):
        assert R is None, "not yet implemented"

        if self.scaled:
            if u is not None:
                u = scale_model_signals(self.model, "u", u)

            y_meas = scale_model_signals(self.model, "y", y_meas)

        if u is None:
            u = np.nan * np.ones((self.model.inputs.shape[0],))

        y_est: np.ndarray = self.f_out(self.x, u).full().reshape((-1,))

        C: np.ndarray = self.C(self.x, u).full()

        K = self.P @ C.T @ np.linalg.inv(C @ self.P @ C.T + self.R)

        self.x = self.x + K @ (y_meas - y_est)
        n = len(self.x)
        self.P = (np.eye(n) - K @ C) @ self.P

    def get_estimate(
        self, include_covariance: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if not include_covariance:
            if self.scaled:
                return unscale_model_signals(self.model, "x", self.x).reshape((-1,))
            else:
                return self.x.copy().reshape((-1,))
        else:
            if self.scaled:
                return (
                    unscale_model_signals(self.model, "x", self.x).reshape((-1,)),
                    _unscale_covariance_matrix(
                        self.model.scalings, self.x_names, self.P
                    ),
                )
            else:
                return (self.x.copy().reshape((-1,)), self.P.copy())
