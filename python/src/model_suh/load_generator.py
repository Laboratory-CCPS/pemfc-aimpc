import random
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import numpy as np


def _I_load_reference(t: float | np.ndarray) -> float | np.ndarray:
    return (
        100.0
        + 80 * (t >= 2)
        + 40 * (t >= 6)
        - 20 * (t >= 10)
        + 60 * (t >= 14)
        + 60 * (t >= 22)
    )


def _I_load_minmax(t: float | np.ndarray) -> float | np.ndarray:
    return (
        100.0
        - 80 * (t >= 2)
        + 300 * (t >= 6)
        - 300 * (t >= 10)
        + 300 * (t >= 14)
        - 220 * (t >= 22)
    )


_PREDEFINED_LOADS: dict[
    str, tuple[tuple[float, float], Callable[[float | np.ndarray], float | np.ndarray]]
] = {
    "reference": ((0.0, 30.0), _I_load_reference),
    "minmax": ((0.0, 25.0), _I_load_minmax),
}


@dataclass
class _GenParams:
    I_init: float
    I_range: tuple[float, float]
    init_duration: float
    load_change_step_prop: float
    load_step_hold_duration_range: float
    load_ramp_duration_range: tuple[float, float]
    load_ramp_hold_prob: float
    load_ramp_hold_duration_range: tuple[float, float]


class LoadGenerator:
    def __init__(
        self,
        *args,
        predefined_extrap: Literal["nan", "cycle", "hold"] = "nan",
        **kwargs,
    ):
        self._predefined_load: Optional[str] = None
        self._gen_params: Optional[_GenParams] = None
        self._gen_state: Optional[tuple[str, float, float, float, float]] = None
        self._predefined_extrap = predefined_extrap

        if (len(args) == 1) and (isinstance(args[0], str)):
            if len(kwargs) > 0:
                raise ValueError("invalid initialization parameters")

            if args[0] not in _PREDEFINED_LOADS.keys():
                raise ValueError(
                    f"unknown predefined load. known load profiles: {', '.join(_PREDEFINED_LOADS.keys())}"
                )

            self._predefined_load = args[0]
        else:
            self._gen_params = _GenParams(**kwargs)
            self._gen_state = (
                "init",
                0.0,
                self._gen_params.I_init,
                self._gen_params.init_duration,
                self._gen_params.I_init,
            )

    def cycle_duration(self) -> Optional[float]:
        if self._predefined_load is not None:
            (t0, t1) = _PREDEFINED_LOADS[self._predefined_load][0]
            return t1 - t0

        return None

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        if self._predefined_load is not None:
            ((t0, t1), fct) = _PREDEFINED_LOADS[self._predefined_load]

            if isinstance(t, np.ndarray):
                if self._predefined_extrap == "cycle":
                    t = (t - t0) % (t1 - t0) + t0
                elif self._predefined_extrap == "hold":
                    t = t.copy()
                    t[t < t0] = t0
                    t[t > t1] = t1

                values = fct(t)

                if self._predefined_extrap == "nan":
                    values[(t < t0) | (t > t1)] = np.nan

                return values

            else:
                if t < t0:
                    if self._predefined_extrap == "nan":
                        return np.nan
                    elif self._predefined_extrap == "cycle":
                        t = (t - t0) % (t1 - t0) + t0
                    else:
                        t = t0
                elif t > t1:
                    if self._predefined_extrap == "nan":
                        return np.nan
                    elif self._predefined_extrap == "cycle":
                        t = (t - t0) % (t1 - t0) + t0
                    else:
                        t = t1

                return fct(t)

        if isinstance(t, np.ndarray):
            I_load = np.zeros_like(t)

            for i, tc in enumerate(t):
                I_load[i] = self(tc)

            return I_load

        while True:
            (_, t0, I0, t1, I1) = self._gen_state

            if t >= t1:
                self._gen_state = _get_next_state(self._gen_state, self._gen_params)
                continue

            I_load = I0 + (t - t0) * (I1 - I0) / (t1 - t0)
            return I_load


def _get_next_state(oldstate: tuple[str, Any], params: _GenParams) -> tuple[str, Any]:

    (state, _, _, t0, I0) = oldstate

    match state:
        case "init" | "hold" | "step":
            if random.uniform(0.0, 1.0) <= params.load_change_step_prop:
                next_state = "step"
            else:
                next_state = "ramp"

        case "ramp":
            if random.uniform(0.0, 1.0) <= params.load_ramp_hold_prob:
                next_state = "hold"
            else:
                if random.uniform(0.0, 1.0) <= params.load_change_step_prop:
                    next_state = "step"
                else:
                    next_state = "ramp"

    if next_state == "step":
        I_target = random.uniform(*params.I_range)
        duration = random.uniform(*params.load_step_hold_duration_range)

        return (next_state, t0, I_target, t0 + duration, I_target)

    elif next_state == "ramp":
        I_target = random.uniform(*params.I_range)
        duration = random.uniform(*params.load_ramp_duration_range)

        return (next_state, t0, I0, t0 + duration, I_target)

    elif next_state == "hold":
        duration = random.uniform(*params.load_ramp_hold_duration_range)

        return (next_state, t0, I0, t0 + duration, I0)

    assert False
