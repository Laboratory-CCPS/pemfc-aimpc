from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from ..ode_modelling.casadi_helper import casadi_vars_to_str
from ..ode_modelling.ode_model import OdeModel

FixedValueArg = float | str


@dataclass
class FixedValue:
    numeric_value: Optional[float]
    parameter_name: Optional[str]

    def is_numeric(self) -> bool:
        return self.numeric_value is not None

    def get_numeric_value(self) -> float:
        return self.numeric_value

    def get_parameter_name(self) -> str:
        return self.parameter_name

    @staticmethod
    def Parameter(name: str):
        return FixedValue(numeric_value=None, parameter_name=name)

    @staticmethod
    def Numeric(value: float):
        return FixedValue(numeric_value=value, parameter_name=None)


@dataclass
class FreeInput:
    signal: str
    min: float
    max: float
    init: float


@dataclass
class FixedInput:
    signal: str
    value: FixedValue


@dataclass
class Constraint:
    signal: str
    min: float
    max: float
    feedthrough: bool


@dataclass
class QuadraticCost:
    signal: str
    ref: FixedValue
    weight_stage: float
    weight_end: float
    scaled_weights: bool
    feedthrough: bool


@dataclass
class QuadraticDiffQuotCost:
    signal: str
    quot: tuple[float]
    ts_order: int
    weight: float
    scaled_weight: bool
    feedthrough: bool
    prev_value: Optional[FixedValue]


@dataclass
class LinearCost:
    signal: str
    weight_stage: bool
    weight_end: bool
    scaled_weights: bool
    feedthrough: bool


@dataclass
class Parameter:
    signal: str
    is_vector: bool
    scaled_like: str


class SignalProp(Protocol):
    signal: str


def _warn(msg: str):
    print(f"Waring: {msg}")


def get_signals(v: Iterable[SignalProp]) -> list[str]:
    return [x.signal for x in v]


def find_signal_idx(v: Iterable[SignalProp], signal: str) -> Optional[int]:
    for i, el in enumerate(v):
        if el.signal == signal:
            return i

    return None


class MpcProblem:
    def __init__(
        self,
        model: OdeModel,
    ):
        self.model = model

        self.u_names = casadi_vars_to_str(self.model.inputs)
        self.x_names = casadi_vars_to_str(self.model.states)
        self.y_names = self.model.y_names

        self.parameters: list[Parameter] = []

        self.free_inputs: list[FreeInput] = []
        self.fixed_inputs: list[str] = []
        self.constraints: list[Constraint] = []
        self.costs_quadratic: list[QuadraticCost] = []
        self.costs_quadratic_diffquot: list[QuadraticDiffQuotCost] = []
        self.costs_linear: list[LinearCost] = []

    @property
    def nu_free(self) -> int:
        return len(self.free_inputs)

    def add_free_input(
        self, name: str, min: float, max: float, init: Optional[float] = None
    ):
        if name not in self.u_names:
            raise ValueError(f"'{name}' is not an input of the model")

        if (idx := find_signal_idx(self.free_inputs, name)) is None:
            self.free_inputs.append(
                FreeInput(name, min, max, np.nan if init is None else init)
            )

        else:
            _warn(
                f"'{name}' was already defined as a free input. "
                "Updating its parameters, but the previous index of the input remains valid."
            )
            self.free_inputs[idx](
                FreeInput(name, min, max, np.nan if init is None else init)
            )

    def add_fixed_input(self, name: str, value: FixedValueArg):
        if name not in self.u_names:
            raise ValueError(f"'{name}' is not an input of the model")

        if find_signal_idx(self.free_inputs, name) is not None:
            _warn(f"'{name}' is already a free input. The fixed value will be ignored.")

        if (idx := find_signal_idx(self.fixed_inputs, name)) is None:
            self.fixed_inputs.append(value)
        else:
            self.fixed_inputs[idx] = value

    def add_fixed_inputs(self, fixed_inputs: dict[str, FixedValueArg]):
        for k, v in fixed_inputs.items():
            self.add_fixed_input(k, v)

    def add_constraint(
        self, name: str, min: float, max: float, feedthrough: Optional[bool] = None
    ):
        if name in self.u_names:
            raise ValueError(
                f"'{name}' is an input. Constraints on inputs must be specified with add_free_input"
            )

        if min > max:
            raise ValueError("invalid constraint: max is smaller than min")

        feedthrough = self._check_signal(name, feedthrough)

        for i, c in enumerate(self.constraints):
            if (name == c.signal) and (feedthrough == c.feedthrough):
                _warn(
                    f"For '{name}' a constraint with the same feedthrough value was already defined and gets overwritten now."
                )

                self.constraints[i] = Constraint(name, min, max, feedthrough)
                self.dirty = True
                return

        self.constraints.append(Constraint(name, min, max, feedthrough))

    def add_quadratic_cost(
        self,
        name: str,
        ref: float | str,
        weight_stage: float = 0.0,
        weight_end: float = 0.0,
        feedthrough: Optional[bool] = None,
        scaled_weights: bool = False,
    ):
        feedthrough = self._check_signal(name, feedthrough)

        ref = self._check_fixed_value(ref, name)
        cost = QuadraticCost(
            name, ref, weight_stage, weight_end, scaled_weights, feedthrough
        )

        for i, c in enumerate(self.costs_quadratic):
            if (name == c.signal) and (feedthrough == c.feedthrough):
                _warn(
                    f"For '{name}' a quadratic cost with the same feedthrough value was already defined and gets overwritten now."
                )

                self.costs_quadratic[i] = cost
                return

        self.costs_quadratic.append(cost)

    def add_quadratic_diffquot_cost(
        self,
        name: str,
        diff_quot: Sequence[float],
        ts_order: Optional[int] = None,
        weight: float = 0.0,
        feedthrough: Optional[bool] = None,
        scaled_weight: bool = False,
        prev_value: Optional[FixedValueArg] = None,
    ):
        feedthrough = self._check_signal(name, feedthrough)

        diff_quot = tuple(diff_quot)

        if not np.isclose(sum(diff_quot), 0):
            _warn(f"difference quotient factors for '{name}' don't sum up to 0.")

        if ts_order is None:
            ts_order = 0.0

        if prev_value is not None:
            prev_value = self._check_fixed_value(prev_value, name)

        cost = QuadraticDiffQuotCost(
            name, diff_quot, ts_order, weight, scaled_weight, feedthrough, prev_value
        )

        for i, c in enumerate(self.costs_quadratic_diffquot):
            if (name == c.signal) and (feedthrough == c.feedthrough):
                _warn(
                    f"For '{name}' a quadratic diff quot cost with the same feedthrough value was already defined and gets overwritten now."
                )

                self.costs_quadratic_diffquot[i] = cost
                return

        self.costs_quadratic_diffquot.append(cost)

    def add_linear_cost(
        self,
        name: str,
        weight_stage: float,
        weight_end: float,
        feedthrough: Optional[bool] = None,
        scaled_weights: bool = False,
    ):
        feedthrough = self._check_signal(name, feedthrough)

        cost = LinearCost(name, weight_stage, weight_end, scaled_weights, feedthrough)

        for i, c in enumerate(self.costs_quadratic):
            if (name == c.signal) and (feedthrough == c.feedthrough):
                self.costs_linear[i] = cost
                return

        self.costs_linear.append(cost)

    def _check_fixed_value(self, value: FixedValueArg, ref_signal: str) -> FixedValue:
        if not isinstance(value, str):
            return FixedValue.Numeric(value)

        name = value.strip()

        if not name.startswith("#"):
            raise ValueError("Fixed value must be numerical or start with '#'")

        name = name[1:]

        is_vector = name.endswith("[]")

        if is_vector:
            name = name[:-2]

        if name in self.x_names:
            raise ValueError(
                f"'{name}' is already a state and cannot be used as parameter name"
            )

        if name in self.u_names:
            raise ValueError(
                f"'{name}' is already an input and cannot be used as parameter name"
            )

        if name in self.y_names:
            raise ValueError(
                f"'{name}' is already an output and cannot be used as parameter name"
            )

        if name in {"x0", "u", "lam_g"}:
            raise ValueError(f"'{name}' is a reserved name")

        if find_signal_idx(self.parameters, name) is None:
            self.parameters.append(Parameter(name, is_vector, ref_signal))
        else:
            raise ValueError(f"'{name}' is already used as a parameter name")

        return FixedValue.Parameter(name)

    def _check_signal(self, name: str, feedthrough: Optional[bool]) -> bool:
        if name in self.u_names:
            if feedthrough is not None:
                raise ValueError(
                    "The parameter 'feedthrough' can only be used with outputs, "
                    f"but '{name}' is an input."
                )

            feedthrough = True

        elif name in self.x_names:
            if feedthrough is not None:
                raise ValueError(
                    "The parameter 'feedthrough' can only be used with outputs, "
                    f"but '{name}' is a state."
                )

            feedthrough = False

        elif name in self.y_names:
            feedthrough = False if feedthrough is None else feedthrough

        else:
            raise ValueError(f"'{name}' is not a signal of the model.")

        return feedthrough

    def get_all_constraints(self) -> dict[str, tuple[float, float]]:
        constraints = {v.signal: (v.min, v.max) for v in self.constraints}
        input_constraints = {v.signal: (v.min, v.max) for v in self.free_inputs}

        constraints.update(input_constraints)

        return constraints

    def check(self) -> bool:
        defined_inputs = set(
            get_signals(self.free_inputs) + get_signals(self.fixed_inputs)
        )

        model_inputs = set(self.u_names)

        missing_inputs = model_inputs - defined_inputs

        if len(missing_inputs) > 0:
            print(
                "the following model inputs are neither defined as free nor fixed inputs: "
                f"{', '.join(missing_inputs)}"
            )
            return False

        return True
