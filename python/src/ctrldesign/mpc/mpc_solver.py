from collections.abc import Sequence
from typing import Any, Literal, Optional

import casadi
import numpy as np

from ..ode_modelling import ode_model
from ..ode_modelling.helper import find_element_idx
from ..ode_modelling.model_helper import scale_data, unscale_data
from ..ode_modelling.scaling import Scaling
from ..ode_modelling.simresult import SimResult
from .mpc_problem import (
    Constraint,
    FixedInput,
    FixedValue,
    FreeInput,
    LinearCost,
    MpcProblem,
    Parameter,
    QuadraticCost,
    QuadraticDiffQuotCost,
    _warn,
    find_signal_idx,
    get_signals,
)

DiscretizatonType = Literal["rk1", "rk4", "idas"]


class MpcSolver:
    def __init__(
        self,
        mpcproblem: MpcProblem,
        Ts: float,
        N: int,
        discretization: DiscretizatonType,
        substeps: int = 1,
        scale_model: bool = False,
        use_lambda: bool = False,
        multiple_shooting: bool = False,
        use_opti_function: bool = False,
        verbose: bool = True,
        plugin_options: Optional[dict[str, Any]] = None,
        solver_options: Optional[dict[str, Any]] = None,
    ):
        assert discretization in ("rk1", "rk4", "idas")

        self.scaled = scale_model
        self.orig_scaled = mpcproblem.model.scaled

        self.scalings = dict[str, Scaling]

        if self.scaled:
            self.model = ode_model.scale_model(mpcproblem.model)
            self.scalings = self.model.scalings.copy()
        else:
            self.model = mpcproblem.model
            self.scalings = {}

        all_names = set(mpcproblem.u_names + mpcproblem.x_names + mpcproblem.y_names)
        missing_scalings = all_names - set(self.scalings.keys())

        for name in missing_scalings:
            self.scalings[name] = Scaling(1, 0)

        for p in mpcproblem.parameters:
            self.scalings[p.signal] = self.scalings[p.scaled_like]

        self.Ts = Ts
        self.N = N

        self.multiple_shooting = multiple_shooting
        self.use_opti_function = use_opti_function
        self.use_lambda = use_lambda

        self.opti = casadi.Opti()

        nu = len(mpcproblem.free_inputs)
        nx = len(mpcproblem.x_names)

        (u_free, xk) = _create_optimization_variables(
            N, nu, nx, multiple_shooting, self.opti
        )

        (x0, params) = _create_optimization_parameters(
            N, nx, mpcproblem.parameters, self.opti
        )

        u = _build_input_matrix(
            mpcproblem.u_names,
            mpcproblem.free_inputs,
            mpcproblem.fixed_inputs,
            mpcproblem.parameters,
            u_free,
            params,
        )

        (x, y, y_ft) = _sim_steps_and_create_ms_constraints(
            self.model, x0, u, Ts, discretization, substeps, xk, self.opti
        )

        _create_input_constraints(
            mpcproblem.free_inputs, self.scalings, u_free, self.opti
        )

        _create_inequality_constraints(
            mpcproblem.constraints,
            self.scalings,
            mpcproblem.x_names,
            mpcproblem.y_names,
            x,
            y,
            y_ft,
            self.opti,
        )

        _create_objective_function(
            mpcproblem.costs_quadratic,
            mpcproblem.costs_quadratic_diffquot,
            mpcproblem.costs_linear,
            self.scalings,
            self.scaled,
            self.Ts,
            mpcproblem.free_inputs,
            mpcproblem.parameters,
            mpcproblem.x_names,
            mpcproblem.y_names,
            x,
            y,
            y_ft,
            u_free,
            params,
            self.opti,
        )

        if verbose:
            print_level = 5
            print_time = 1
        else:
            print_level = 0
            print_time = 0

        if plugin_options is None:
            plugin_options = {}
        else:
            plugin_options = plugin_options.copy()

        if solver_options is None:
            solver_options = {}
        else:
            solver_options = solver_options.copy()

        plugin_options.setdefault("print_time", print_time)
        plugin_options.setdefault("detect_simple_bounds", True)

        solver_options.setdefault("print_level", print_level)

        self.opti.solver("ipopt", plugin_options, solver_options)

        if not self.use_opti_function:
            self.f = None
        else:
            xret = x[:, 1:]
            ps = [params[p.signal] for p in mpcproblem.parameters]

            if multiple_shooting:
                if use_lambda:
                    self.f = self.opti.to_function(
                        "f",
                        [x0, u_free, xk, *ps, self.opti.lam_g],
                        [u_free, xret, self.opti.lam_g],
                    )
                else:
                    self.f = self.opti.to_function(
                        "f",
                        [x0, u_free, xk, *ps],
                        [u_free, xret],
                    )
            else:
                if use_lambda:
                    self.f = self.opti.to_function(
                        "f",
                        [x0, u_free, *ps, self.opti.lam_g],
                        [u_free, xret, self.opti.lam_g],
                    )
                else:
                    self.f = self.opti.to_function(
                        "f",
                        [x0, u_free, *ps],
                        [u_free, xret],
                    )

        self._casadi_vars = {
            "x0": x0,
            "u_free": u_free,
            "x1_N": x[:, 1:],
            "xk": xk,
            "params": params,
        }

        self.free_inputs = mpcproblem.free_inputs
        self.fixed_inputs = mpcproblem.fixed_inputs
        self.parameters = mpcproblem.parameters

        self.x_names = mpcproblem.x_names
        self.u_names = mpcproblem.u_names
        self.y_names = mpcproblem.y_names

        self.mpcproblem = mpcproblem

    def get_all_constraints(self) -> dict[str, tuple[float, float]]:
        return self.mpcproblem.get_all_constraints()

    def solve_step(
        self, x0: np.ndarray, u_init: np.ndarray, **params
    ) -> dict[str, Any]:

        if len(x0.shape) == 1:
            x0 = x0.reshape((-1, 1))

        step_result = {"x0": x0[:, [0]]}

        if self.scaled:
            x0 = scale_data(self.scalings, self.x_names, x0)
            u_init = scale_data(self.scalings, get_signals(self.free_inputs), u_init)

        lam_g = params.get("lam_g", None)

        if (not self.use_lambda) and (lam_g is not None):
            raise ValueError("'lam_g' must be None")
        elif self.use_lambda and (lam_g is None):
            lam_g = np.zeros(self.opti.lam_g.shape)

        param_values: list[np.ndarray | float] = []

        for p in self.parameters:
            pname = p.signal

            if pname not in params:
                raise ValueError(f"missing parameter '{pname}' in kwargs")

            value = params[pname]

            if (
                (not p.is_vector)
                and (isinstance(value, np.ndarray))
                and (len(value) != 1)
            ):
                raise ValueError(f"parameter '{pname}' must be a scalar")
            elif p.is_vector:
                if (not isinstance(value, np.ndarray)) or (len(value) == 1):
                    value = value * np.ones((self.N + 1,))
                elif len(value) != self.N + 1:
                    raise ValueError(
                        f"parameter '{pname}' must be a {self.N + 1}-dimensional (N+1-dimensional) vector"
                    )

            param_values.append(value)

        keys = set(params.keys())
        keys = keys - set(p.signal for p in self.parameters)
        keys = keys - {"lam_g"}

        if len(keys) > 0:
            raise ValueError(f"unknown arguments: {', '.join(keys)}")

        init_args = [u_init]

        if self.multiple_shooting:
            if x0.shape[1] == 1:
                x_init = np.repeat(x0, self.N, axis=1)
            else:
                x_init = x0[:, 1:]
                x0 = x0[:, [0]]

            init_args.append(x_init)
        else:
            x0 = x0[:, [0]]

        if self.use_opti_function:
            if self.use_lambda:
                (u_opt, x_opt, lam_g) = self.f(x0, *init_args, *param_values, lam_g)

                u_opt = u_opt.full()
                x_opt = x_opt.full()
            else:
                (u_opt, x_opt) = self.f(x0, *init_args, *param_values)

                u_opt = u_opt.full()
                x_opt = x_opt.full()
        else:
            self.opti.set_value(self._casadi_vars["x0"], x0)

            for i, p in enumerate(self.parameters):
                self.opti.set_value(
                    self._casadi_vars["params"][p.signal], param_values[i]
                )

            self.opti.set_initial(self._casadi_vars["u_free"], u_init)

            if self.multiple_shooting:
                self.opti.set_initial(self._casadi_vars["xk"], x_init)

            if self.use_lambda:
                self.opti.set_initial(self.opti.lam_g, lam_g)

            sol: casadi.OptiSol = self.opti.solve_limited()

            u_opt = sol.value(self._casadi_vars["u_free"])
            x_opt = sol.value(self._casadi_vars["x1_N"])

            if self.use_lambda:
                lam_g = sol.value(self.opti.lam_g)

            step_result["opti_sol"] = sol

        if self.scaled:
            step_result["u_opt"] = unscale_data(
                self.scalings, get_signals(self.free_inputs), u_opt
            )
            step_result["x_opt"] = unscale_data(self.scalings, self.x_names, x_opt)
        else:
            step_result["u_opt"] = u_opt
            step_result["x_opt"] = x_opt

        if self.use_lambda:
            step_result["lam_g"] = lam_g
        else:
            step_result["lam_g"] = None

        step_result["u_fixed"] = _get_u_fixed(
            self.fixed_inputs, self.parameters, param_values, self.N
        )

        step_result["params"] = {
            p.signal: pvalue for (p, pvalue) in zip(self.parameters, param_values)
        }

        return step_result

    def get_u_init(
        self,
        u_init: Optional[np.ndarray] = None,
        **kwargs: dict[str, float | np.ndarray],
    ) -> np.ndarray:
        n_u_free = len(self.free_inputs)

        u0_provided = u_init is not None

        if u0_provided:
            u_0 = u_init.copy()
        else:
            u_0 = np.zeros((n_u_free, self.N))

        provided_inputs = set(kwargs.keys()) - {"u_init"}
        expected_inputs = set(get_signals(self.free_inputs))

        unknown_inputs = provided_inputs - expected_inputs

        if len(unknown_inputs) > 0:
            raise ValueError(
                f"the following parameters aren't inputs: {', '.join(sorted(unknown_inputs))}"
            )

        for i in range(n_u_free):
            uname = self.free_inputs[i].signal

            if uname in kwargs:
                u_0[i, :] = kwargs[uname]
            elif not u0_provided:
                u_0[i, :] = self.free_inputs[i].init

        return u_0

    def get_complete_input(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "use 'construct_model_input' instead of 'get_complete_input'",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.construct_model_input(*args, **kwargs)

    def construct_model_input(
        self,
        u_free: np.ndarray,
        u_fixed: Optional[np.ndarray] = None,
        oversampling_rate: int = 1,
    ) -> np.ndarray:
        n = u_free.shape[1]

        u = np.zeros((len(self.u_names), n * oversampling_rate))

        constant_fixed: Optional[bool]

        if u_fixed is None:
            if len(self.fixed_inputs) > 0:
                raise ValueError("the MPC has fixed inputs, 'u_fixed' must be provided")

            constant_fixed = None

        elif (len(u_fixed.shape) == 1) or (u_fixed.shape[1] == 1):
            u_fixed = u_fixed.reshape((-1,))
            constant_fixed = True
        else:
            constant_fixed = False

        indices = _get_input_indices(self.u_names, self.free_inputs, self.fixed_inputs)

        idx1 = 0
        for k in range(n):
            idx0 = idx1
            idx1 = idx0 + oversampling_rate

            for idx_u, (kind, idx_source) in enumerate(indices):
                if kind == "free":
                    u[idx_u, idx0:idx1] = u_free[idx_source, k]
                else:
                    assert constant_fixed is not None

                    if constant_fixed:
                        u[idx_u, idx0:idx1] = u_fixed[idx_source]
                    else:
                        u[idx_u, idx0:idx1] = u_fixed[idx_source, k]

        return u

    def get_next_initial_values(
        self, step_result: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        u_init = np.hstack((step_result["u_opt"][:, 1:], step_result["u_opt"][:, -1:]))

        if self.multiple_shooting:
            x_init = np.hstack((step_result["x_opt"], step_result["x_opt"][:, [-1]]))
        else:
            x_init = step_result["x_opt"][:, 0]

        lam_g = step_result["lam_g"]

        return (x_init, u_init, lam_g)

    def get_signal_value(
        self,
        name: str,
        k: Optional[int] = None,
        x: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
    ) -> float:

        if (x is not None) and (
            idx := find_element_idx(self.x_names, name)
        ) is not None:
            if k is None:
                if len(x.shape) == 1:
                    return x[idx]
                else:
                    return x[idx, 0]

            return x[idx, k]

        if (u is not None) and (
            idx := find_element_idx(self.u_names, name)
        ) is not None:
            if k is None:
                if len(u.shape) == 1:
                    return u[idx]
                else:
                    return u[idx, 0]

            return u[idx, k]

        raise NotImplementedError()

    def _extract_model_signal_values(
        self,
        name: str,
        x: Optional[np.ndarray],
        u: Optional[np.ndarray],
    ) -> float | np.ndarray:
        if (idx := find_element_idx(self.x_names, name)) is not None:
            if x is None:
                raise ValueError(
                    f"'{name}' is a state, but no state values are provided"
                )

            if len(x.shape) == 1:
                return x[idx]
            else:
                return x[idx, :]

        if (idx := find_element_idx(self.u_names, name)) is not None:
            if u is None:
                raise ValueError(
                    f"'{name}' is an input, but no input values are provided"
                )

            if len(u.shape) == 1:
                return u[idx]
            else:
                return u[idx, :]

        raise NotImplementedError()

    def extract_model_signal_values(
        self,
        names: str | Sequence[str],
        x: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        return_tuple: bool = False,
    ) -> float | np.ndarray | tuple[float, ...] | tuple[np.ndarray]:

        if isinstance(names, str):
            names = [names]

        values: list[np.ndarray | float] = []

        for name in names:
            values.append(self._extract_model_signal_values(name, x, u))

        if return_tuple:
            return tuple(values)

        else:
            return np.vstack(values)

    def _extract_mpc_input_values(
        self,
        name: str,
        u_free: Optional[np.ndarray],
        u_fixed: Optional[np.ndarray],
    ) -> float | np.ndarray:
        if (idx := find_element_idx(get_signals(self.free_inputs), name)) is not None:
            if u_free is None:
                raise ValueError(
                    f"'{name}' is a free input, but no free input values are provided"
                )

            if len(u_free.shape) == 1:
                return u_free[idx]
            else:
                return u_free[idx, :]

        if (idx := find_element_idx(get_signals(self.fixed_inputs), name)) is not None:
            if u_fixed is None:
                raise ValueError(
                    f"'{name}' is a fixed input, but no fixed input values are provided"
                )

            if len(u_fixed.shape) == 1:
                return u_fixed[idx]
            else:
                return u_fixed[idx, :]

        raise ValueError(f"'{name}' is neither a free nor a fixed input")

    def extract_mpc_input_values(
        self,
        names: str | Sequence[str],
        u_free: Optional[np.ndarray] = None,
        u_fixed: Optional[np.ndarray] = None,
        return_tuple: bool = False,
    ) -> float | np.ndarray | tuple[float, ...] | tuple[np.ndarray]:

        if isinstance(names, str):
            names = [names]

        values: list[np.ndarray | float] = []

        for name in names:
            values.append(self._extract_mpc_input_values(name, u_free, u_fixed))

        if return_tuple:
            return tuple(values)

        else:
            return np.vstack(values)

    def sol_into_simresult(self, mpc_sol: dict[str, Any]) -> SimResult:

        t = np.arange(0, self.N + 1) * self.Ts
        x = np.hstack((mpc_sol["x0"], mpc_sol["x_opt"]))

        u = self.construct_model_input(mpc_sol["u_opt"], mpc_sol["u_fixed"], 1)
        u = np.hstack((u, np.nan * np.zeros((u.shape[0], 1))))

        f_out = casadi.Function(
            "outputs", [self.model.states, self.model.inputs], [self.model.y]
        )

        if self.scaled:
            x_sim = scale_data(self.scalings, self.x_names, x)
            u_sim = scale_data(self.scalings, self.u_names, u)
        else:
            x_sim = x
            u_sim = u

        y = np.zeros((len(self.y_names), self.N + 1))

        for i in range(self.N + 1):
            if i == 0:
                ui = u_sim[:, 0]
            else:
                ui = u_sim[:, i - 1]

            y[:, [i]] = f_out(x_sim[:, i], ui).full()

        y_ft = np.zeros((len(self.y_names), self.N + 1))

        for i in range(self.N + 1):
            y_ft[:, [i]] = f_out(x_sim[:, i], u_sim[:, i]).full()

        if self.scaled:
            y = unscale_data(self.scalings, self.y_names, y)
            y_ft = unscale_data(self.scalings, self.y_names, y_ft)

        return SimResult(
            t=t,
            u=u,
            x=x,
            y=y,
            y_ft=y_ft,
            u_names=self.u_names,
            x_names=self.x_names,
            y_names=self.y_names,
            scalings=self.scalings,
            scaled=self.orig_scaled,
            constraints=self.mpcproblem.get_all_constraints(),
            discrete_model=False,
            Ts=self.Ts,
            desc="",
        )


def _get_expr(
    name: str,
    x_names: Sequence[str],
    y_names: Sequence[str],
    x: casadi.MX,
    y: casadi.MX,
) -> tuple[casadi.MX, Literal["state", "output"]]:
    if (idx := find_element_idx(x_names, name)) is not None:
        return (x[idx, :], "state")

    if (idx := find_element_idx(y_names, name)) is not None:
        return (y[idx, :], "output")

    assert False


def _get_u_fixed(
    fixed_inputs: Sequence[FixedInput],
    parameters: Sequence[Parameter],
    param_values: Sequence[np.ndarray],
    N: int,
) -> np.ndarray:
    u = np.zeros((len(fixed_inputs), N))

    for i, f in enumerate(fixed_inputs):
        if f.value.is_numeric():
            u[i, :] = f.value.get_numeric_value()
            continue

        pname = f.value.get_parameter_name()

        idx = find_signal_idx(parameters, pname)

        value = param_values[idx]

        if len(value) == 1:
            u[i, :] = value
        else:
            u[i, :] = value[:-1]

    return u


def _create_optimization_variables(
    N_steps: int,
    n_inputs: int,
    n_states: int,
    multiple_shooting: bool,
    opti: casadi.Opti,
) -> tuple[casadi.MX, Optional[casadi.MX]]:
    # the rows of u correspond to the free variables in the order
    # in which they were added by add_free_input, i.e. the order
    # is given by the order of the fields of self.u_free_inputs
    u_free = opti.variable(n_inputs, N_steps)

    xk: Optional[casadi.MX] = None

    if multiple_shooting:
        xk = opti.variable(n_states, N_steps)

    return (u_free, xk)


def _create_optimization_parameters(
    N_steps: int, n_states: int, parameters: Sequence[Parameter], opti: casadi.Opti
) -> tuple[casadi.MX, dict[str, casadi.MX]]:
    x0 = opti.parameter(n_states)

    params = {}
    for p in parameters:

        if not p.is_vector:
            pdim = 1
        else:
            # For now, we just create for each vectorial parameters N + 1 entries.
            # (It might not be used all, depending of how the parameter is used,
            # but we leave this optimization for later. It should not influence
            # the solving at all but just add a minor overhead in calling the
            # solver.)
            pdim = N_steps + 1

        params[p.signal] = opti.parameter(pdim)

    return (x0, params)


def _build_input_matrix(
    u_names: tuple[str, ...],
    free_inputs: list[FreeInput],
    fixed_inputs: Sequence[FixedInput],
    parameters: Sequence[Parameter],
    u_free: casadi.MX,
    params: dict[str, casadi.MX],
) -> casadi.MX:
    N = u_free.shape[1]

    indices = _get_input_indices(u_names, free_inputs, fixed_inputs)

    u = []

    for k in range(N):
        uk = []

        for source, idx in indices:
            if source == "free":
                uk.append(u_free[idx, k])
            else:
                uk.append(
                    _get_fixed_value(fixed_inputs[idx].value, k, parameters, params)
                )

        u.append(casadi.vertcat(*uk))

    return casadi.horzcat(*u)


def _get_input_indices(
    u_names: tuple[str, ...],
    free_inputs: list[FreeInput],
    fixed_inputs: Sequence[FixedInput],
) -> list[tuple[Literal["free", "fixed"], int]]:
    indices = []
    missing_inputs = []

    for name in u_names:
        if (idx := find_signal_idx(free_inputs, name)) is not None:
            indices.append(("free", idx))
            continue

        if (idx := find_signal_idx(fixed_inputs, name)) is not None:
            indices.append(("fixed", idx))
            continue

        missing_inputs.append(name)

    if len(missing_inputs) > 0:
        raise ValueError(
            "the following input(s) are neither specified as free or as fixed inputs: "
            f"{', '.join(missing_inputs)}"
        )

    return indices


def _get_fixed_value(
    fv: FixedValue,
    idx: int,
    parameters: Sequence[Parameter],
    params: dict[str, casadi.MX],
) -> float | casadi.MX:
    if fv.is_numeric():
        return fv.get_numeric_value()

    pname = fv.get_parameter_name()

    pidx = find_signal_idx(parameters, pname)

    if parameters[pidx].is_vector:
        return params[pname][idx]
    else:
        return params[pname]


def _create_input_constraints(
    free_inputs: Sequence[FreeInput],
    scalings: Sequence[Scaling],
    u_free: casadi.MX,
    opti: casadi.Opti,
):
    N = u_free.shape[1]

    for idx, input in enumerate(free_inputs):
        name = input.signal
        scaling = scalings[name]

        minval = scaling.scale(input.min)
        maxval = scaling.scale(input.max)

        for k in range(N):
            match (not np.isinf(minval), not np.isinf(maxval)):
                case (True, True):
                    opti.subject_to(opti.bounded(minval, u_free[idx, k], maxval))
                case (True, False):
                    opti.subject_to(minval <= u_free[idx, k])
                case (False, True):
                    opti.subject_to(u_free[idx, k] <= maxval)


def _create_inequality_constraints(
    constraints: Sequence[Constraint],
    scalings: Sequence[Scaling],
    x_names: Sequence[str],
    y_names: Sequence[str],
    x: casadi.MX,
    y: casadi.MX,
    y_ft: casadi.MX,
    opti: casadi.Opti,
):
    N = x.shape[1] - 1

    for constraint in constraints:
        name = constraint.signal

        scaling = scalings[name]

        minval = scaling.scale(constraint.min)
        maxval = scaling.scale(constraint.max)

        if constraint.feedthrough:
            (expr, kind) = _get_expr(name, x_names, y_names, x, y_ft)
        else:
            (expr, kind) = _get_expr(name, x_names, y_names, x, y)

        if (kind == "output") and constraint.feedthrough:
            k0 = 0
            kend = N
        else:
            k0 = 1
            kend = N + 1

        for k in range(k0, kend):
            match (not np.isinf(minval), not np.isinf(maxval)):
                case (True, True):
                    opti.subject_to(opti.bounded(minval, expr[k], maxval))
                case (True, False):
                    opti.subject_to(minval <= expr[k])
                case (False, True):
                    opti.subject_to(expr[k] <= maxval)


def _create_objective_function(
    costs_quadratic: list[QuadraticCost],
    costs_quadratic_diffquot: list[QuadraticDiffQuotCost],
    costs_linear: list[LinearCost],
    scalings: Sequence[Scaling],
    scaled: bool,
    Ts: float,
    free_inputs: Sequence[FreeInput],
    parameters: Sequence[Parameter],
    x_names: Sequence[str],
    y_names: Sequence[str],
    x: casadi.MX,
    y: casadi.MX,
    y_ft: casadi.MX,
    u_free: casadi.MX,
    params: dict[str, casadi.MX],
    opti: casadi.Opti,
):
    J: casadi.MX = 0

    for cost in costs_quadratic:
        J += _get_quadratic_cost_term(
            cost,
            scalings,
            scaled,
            free_inputs,
            parameters,
            x_names,
            y_names,
            x,
            y,
            y_ft,
            u_free,
            params,
        )

    for cost in costs_quadratic_diffquot:
        J += _get_quadratic_dq_cost_term(
            cost,
            scalings,
            scaled,
            Ts,
            free_inputs,
            parameters,
            x_names,
            y_names,
            x,
            y,
            y_ft,
            u_free,
            params,
        )

    for cost in costs_linear:
        J += _get_linear_cost_term(
            cost,
            scalings,
            scaled,
            free_inputs,
            x_names,
            y_names,
            x,
            y,
            y_ft,
            u_free,
        )

    opti.minimize(J)


def _get_quadratic_cost_term(
    cost: QuadraticCost,
    scalings: Sequence[Scaling],
    scaled: bool,
    free_inputs: Sequence[FreeInput],
    parameters: Sequence[Parameter],
    x_names: Sequence[str],
    y_names: Sequence[str],
    x: casadi.MX,
    y: casadi.MX,
    y_ft: casadi.MX,
    u_free: casadi.MX,
    params: dict[str, casadi.MX],
) -> casadi.MX:
    signal = cost.signal
    scaling = scalings[signal]

    N = u_free.shape[1]
    J: casadi.MX = 0

    if (idx := find_signal_idx(free_inputs, signal)) is not None:
        if cost.weight_stage > 0:
            weight = cost.weight_stage

            if (not cost.scaled_weights) and scaled:
                weight *= scaling.factor**2
            elif cost.scaled_weights and (not scaled):
                _warn(
                    "Weight is marked as scaled, but mpc task is not scaled. No (un)scaling is applied."
                )

            for k in range(0, N):
                ref_k = _get_fixed_value(cost.ref, k, parameters, params)
                ref_k = scaling.scale(ref_k)

                J += weight * (u_free[idx, k] - ref_k) ** 2

        if cost.weight_end > 0:
            _warn(f"ignoring end cost for input signal {signal}")

    else:
        if cost.feedthrough:
            (expr, kind) = _get_expr(signal, x_names, y_names, x, y_ft)
        else:
            (expr, kind) = _get_expr(signal, x_names, y_names, x, y)

        if (kind == "output") and cost.feedthrough:
            k0 = 0
            kend = N
        else:
            k0 = 1
            kend = N

        if cost.weight_stage > 0:
            weight = cost.weight_stage

            if (not cost.scaled_weights) and scaled:
                weight *= scaling.factor**2
            elif cost.scaled_weights and (not scaled):
                _warn(
                    "Weight is marked as scaled, but mpc task is not scaled. No (un)scaling is applied."
                )

            for k in range(k0, kend):
                ref_k = _get_fixed_value(cost.ref, k, parameters, params)
                ref_k = scaling.scale(ref_k)

                J += weight * (expr[k] - ref_k) ** 2

        if cost.weight_end > 0:
            weight = cost.weight_end

            if (not cost.scaled_weights) and scaled:
                weight *= scaling.factor**2
            elif cost.scaled_weights and (not scaled):
                _warn(
                    "Weight is marked as scaled, but mpc task is not scaled. No (un)scaling is applied."
                )

            ref_N = _get_fixed_value(cost.ref, N, parameters, params)
            ref_N = scaling.scale(ref_N)

            J += weight * (expr[N] - ref_N) ** 2

    return J


def _get_quadratic_dq_cost_term(
    cost: QuadraticDiffQuotCost,
    scalings: Sequence[Scaling],
    scaled: bool,
    Ts: float,
    free_inputs: Sequence[FreeInput],
    parameters: Sequence[Parameter],
    x_names: Sequence[str],
    y_names: Sequence[str],
    x: casadi.MX,
    y: casadi.MX,
    y_ft: casadi.MX,
    u_free: casadi.MX,
    params: dict[str, casadi.MX],
) -> casadi.MX:
    signal = cost.signal
    scaling = scalings[signal]

    N = u_free.shape[1]
    J: casadi.MX = 0

    weight = cost.weight

    if (not cost.scaled_weight) and scaled:
        weight *= scaling.factor**2
    elif cost.scaled_weight and (not scaled):
        _warn(
            "Weight is marked as scaled, but mpc task is not scaled. No (un)scaling is applied."
        )

    if cost.ts_order != 0:
        weight *= 1 / Ts**cost.ts_order

    if cost.prev_value is None:
        k0 = 0
    else:
        k0 = -1
        prev_value = _get_fixed_value(cost.prev_value, 0, parameters, params)
        prev_value = scaling.scale(prev_value)

    if (idx := find_signal_idx(free_inputs, signal)) is not None:
        kend = N - len(cost.quot) + 1

        for k in range(k0, kend):
            dq = 0

            for i, coeff in enumerate(cost.quot):
                if k == -1:
                    dq += coeff * prev_value
                else:
                    dq += coeff * u_free[idx, k + i]

            J += weight * dq**2

    else:
        if cost.feedthrough:
            (expr, kind) = _get_expr(signal, x_names, y_names, x, y_ft)
        else:
            (expr, kind) = _get_expr(signal, x_names, y_names, x, y)

        if (kind == "output") and cost.feedthrough:
            kend = N - len(cost.quot) + 1
        else:
            # in the case of diff quots it is sensible to start at k=0 even in cases
            # with no feedthrough
            kend = N + 1 - len(cost.quot) + 1

        for k in range(k0, kend):

            for i, coeff in enumerate(cost.quot):
                if k == -1:
                    dq += coeff * prev_value
                else:
                    dq += coeff * expr[k + i]

            J += weight * dq**2

    return J


def _get_linear_cost_term(
    cost: LinearCost,
    scalings: Sequence[Scaling],
    scaled: bool,
    free_inputs: Sequence[FreeInput],
    x_names: Sequence[str],
    y_names: Sequence[str],
    x: casadi.MX,
    y: casadi.MX,
    y_ft: casadi.MX,
    u_free: casadi.MX,
) -> casadi.MX:
    signal = cost.signal
    scaling = scalings[signal]

    N = u_free.shape[1]
    J: casadi.MX = 0

    if (idx := find_signal_idx(free_inputs, signal)) is not None:
        if cost.weight_stage != 0:
            weight = cost.weight_stage

            if (not cost.scaled_weights) and scaled:
                weight *= scaling.factor
            elif cost.scaled_weights and (not scaled):
                _warn(
                    "Weight is marked as scaled, but mpc task is not scaled. No (un)scaling is applied."
                )

            for k in range(0, N):
                J += weight * u_free[idx, k]

        if cost.weight_end != 0:
            _warn(f"ignoring end cost for input signal {signal}")

    else:
        if cost.feedthrough:
            (expr, kind) = _get_expr(signal, x_names, y_names, x, y_ft)
        else:
            (expr, kind) = _get_expr(signal, x_names, y_names, x, y)

        if (kind == "output") and cost.feedthrough:
            k0 = 0
            kend = N
        else:
            k0 = 1
            kend = N

        # weights for linear costs may be negative
        if cost.weight_stage != 0:
            weight = cost.weight_stage

            if (not cost.scaled_weights) and scaled:
                weight *= scaling.factor
            elif cost.scaled_weights and (not scaled):
                _warn(
                    "Weight is marked as scaled, but mpc task is not scaled. No (un)scaling is applied."
                )

            for k in range(k0, kend):
                J += weight * expr[k]

        if cost.weight_end != 0:
            weight = cost.weight_end

            if (not cost.scaled_weights) and scaled:
                weight *= scaling.factor
            elif cost.scaled_weights and (not scaled):
                _warn(
                    "Weight is marked as scaled, but mpc task is not scaled. No (un)scaling is applied."
                )

            J += weight * expr[N]

    return J


def _sim_steps_and_create_ms_constraints(
    model: ode_model.OdeModel,
    x0: casadi.MX,
    u: casadi.MX,
    Ts: float,
    discretization: Literal["rk1", "rk4", "idas"],
    substeps: int,
    xk: Optional[casadi.MX],
    opti: casadi.Opti,
):
    multiple_shooting = xk is not None

    N = u.shape[1]

    x: list[casadi.MX] = [x0]

    if discretization == "idas":
        dae = {"x": model.states, "p": model.inputs, "ode": model.dx}
        F = casadi.integrator("F", "idas", dae, t0=0, tend=Ts / substeps)
    else:
        f_dx = casadi.Function("f", [model.states, model.inputs], [model.dx])

    f_out = casadi.Function("outputs", [model.states, model.inputs], [model.y])

    y: list[casadi.MX] = [f_out(x0, u[:, 0])]
    y_ft: list[casadi.MX] = [f_out(x0, u[:, 0])]

    for i in range(N):
        if discretization == "rk1":
            x_i = _rk1_step(f_dx, x[i], u[:, i], Ts, substeps)
        elif discretization == "rk4":
            x_i = _rk4_step(f_dx, x[i], u[:, i], Ts, substeps)
        elif discretization == "idas":
            x_i = _idas_step(F, x[i], u[:, i], substeps)
        else:
            assert False

        if multiple_shooting:
            x.append(xk[:, i])
            opti.subject_to(xk[:, i] == x_i)
        else:
            x.append(x_i)

        y.append(f_out(x[-1], u[:, i]))

        if i < N - 1:
            y_ft.append(f_out(x[-1], u[:, i + 1]))
        else:
            y_ft.append(y[-1])

    return (casadi.horzcat(*x), casadi.horzcat(*y), casadi.horzcat(*y_ft))


def _rk1_step(
    f: casadi.Function, x0: casadi.MX, u: casadi.MX, Ts: float, substeps: int = 1
) -> casadi.MX:
    dt = Ts / substeps

    x = x0

    for _ in range(substeps):
        x += dt * f(x, u)

    return x


def _rk4_step(
    f: casadi.Function, x0: casadi.MX, u: casadi.MX, Ts: float, substeps: int = 1
) -> casadi.MX:
    dt = Ts / substeps

    x = x0

    for _ in range(substeps):
        k1 = f(x, u)
        k2 = f(x + dt / 2 * k1, u)
        k3 = f(x + dt / 2 * k2, u)
        k4 = f(x + dt * k3, u)
        x += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x


def _idas_step(
    F: casadi.Function, x0: casadi.MX, u: casadi.MX, substeps: int = 1
) -> casadi.MX:
    x = x0

    for _ in range(substeps):
        r = F(x0=x, p=u)
        x = r["xf"]

    return x
