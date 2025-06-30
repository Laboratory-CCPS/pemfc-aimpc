from collections.abc import Sequence
from typing import Literal

import numpy as np

from .casadi_helper import casadi_vars_to_str
from .ode_model import OdeModel
from .scaling import Scaling


def scale_data(
    scalings: dict[str, Scaling], names: Sequence[str], data: np.ndarray, axis: int = 1
) -> np.ndarray:
    if len(data.shape) == 1:
        assert len(names) == data.shape[0]
    elif axis == 1:
        assert len(names) == data.shape[0]
    elif axis == 0:
        assert len(names) == data.shape[1]
    else:
        raise ValueError("'axis' must be 0 or 1")

    data = data.copy()

    for i, name in enumerate(names):
        if name not in scalings.keys():
            print(f"no scaling for signal '{name}' given")
            continue

        if len(data.shape) == 1:
            data[i] = scalings[name].scale(data[i])
        else:
            if axis == 1:
                data[i, :] = scalings[name].scale(data[i, :])
            elif axis == 0:
                data[:, i] = scalings[name].scale(data[:, i])

    return data


def unscale_data(
    scalings: dict[str, Scaling], names: Sequence[str], data: np.ndarray, axis: int = 1
) -> np.ndarray:
    if len(data.shape) == 1:
        assert len(names) == data.shape[0]
    elif axis == 1:
        assert len(names) == data.shape[0]
    elif axis == 0:
        assert len(names) == data.shape[1]
    else:
        raise ValueError("'axis' must be 0 or 1")

    data = data.copy()

    for i, name in enumerate(names):
        if name not in scalings.keys():
            print(f"no scaling for signal '{name}' given")
            continue

        if len(data.shape) == 1:
            data[i] = scalings[name].unscale(data[i])
        else:
            if axis == 1:
                data[i, :] = scalings[name].unscale(data[i, :])
            elif axis == 0:
                data[:, i] = scalings[name].unscale(data[:, i])
            else:
                raise ValueError("'axis' must be 0 or 1")

    return data


def scale_model_signals(
    model: OdeModel,
    kind: Literal["states", "x", "inputs", "u", "outputs", "y"],
    data: np.ndarray,
) -> np.ndarray:
    if kind in ("states", "x"):
        return scale_data(model.scalings, casadi_vars_to_str(model.states), data)
    elif kind in ("inputs", "u"):
        return scale_data(model.scalings, casadi_vars_to_str(model.inputs), data)
    elif kind in ("ouputs", "y"):
        return scale_data(model.scalings, model.y_names, data)

    assert False


def unscale_model_signals(
    model: OdeModel,
    kind: Literal["states", "x", "inputs", "u", "outputs", "y"],
    data: np.ndarray,
) -> np.ndarray:
    if kind in ("states", "x"):
        return unscale_data(model.scalings, casadi_vars_to_str(model.states), data)
    elif kind in ("inputs", "u"):
        return unscale_data(model.scalings, casadi_vars_to_str(model.inputs), data)
    elif kind in ("ouputs", "y"):
        return unscale_data(model.scalings, model.y_names, data)

    assert False
