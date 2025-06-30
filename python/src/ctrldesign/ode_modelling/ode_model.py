from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import casadi
import numpy as np

from .casadi_helper import casadi_vars_to_str, create_casadi_vars
from .helper import find_element_idx
from .scaling import Scaling

ModelFct = Callable[[Optional[tuple[dict[str, Any], dict[str, Any], Any]]], Any]


@dataclass
class OdeModel:
    states: casadi.MX
    inputs: casadi.MX
    dx: casadi.MX
    y: casadi.MX
    y_names: tuple[str, ...]
    scalings: dict[str, Scaling]
    scaled: bool


def get_model(model: ModelFct, p):

    (vars, scalings) = model()

    x = create_casadi_vars(vars["states"])
    u = create_casadi_vars(vars["inputs"])

    states = casadi.vertcat(*x.values())
    inputs = casadi.vertcat(*u.values())

    (dx, y) = model((x, u, p))

    dx = casadi.vertcat(*(dx[state] for state in x.keys()))

    y_names = tuple(y.keys())
    y = casadi.vertcat(*y.values())

    return OdeModel(
        states=states,
        inputs=inputs,
        dx=dx,
        y=y,
        y_names=y_names,
        scalings=scalings,
        scaled=False,
    )


def set_outputs(model: OdeModel, outputs: Sequence[str]) -> OdeModel:
    """Replaces the system outputs

    The new outputs must be chosen from the set of already existing outputs and the
    states.
    """
    new_y_names = list(outputs)
    new_y = []

    x_names = casadi_vars_to_str(model.states)

    for output in new_y_names:
        if (idx := find_element_idx(x_names, output)) is not None:
            new_y.append(model.states[idx])
            continue

        if (idx := find_element_idx(model.y_names, output)) is not None:
            new_y.append(model.y[idx])
            continue

        raise KeyError(f"{output} is neither a state nor an existing output")

    new_y = casadi.vertcat(*new_y)

    return OdeModel(
        states=model.states,
        inputs=model.inputs,
        dx=model.dx,
        y=new_y,
        y_names=new_y_names,
        scalings=model.scalings,
        scaled=model.scaled,
    )


def scale_model(model: OdeModel) -> OdeModel:
    if model.scaled:
        print("model already scaled")
        return model

    return _scale(model, "scale")


def unscale_model(model: OdeModel) -> OdeModel:
    if not model.scaled:
        print("model already unscaled")
        return model

    return _scale(model, "unscale")


def _scale(model: OdeModel, direction: Literal["scale", "unscale"]) -> OdeModel:

    if len(model.scalings) == 0:
        raise ValueError("model doesn't provide scalings")

    scalings = model.scalings

    scaled_names = [v for v in scalings.keys()]

    dx = casadi.MX(model.dx)
    y = casadi.MX(model.y)

    x_names = casadi_vars_to_str(model.states)
    u_names = casadi_vars_to_str(model.inputs)
    y_names = model.y_names

    for i, name in enumerate(x_names):
        if name in scaled_names:
            if direction == "scale":
                dx[i] = scalings[name].scale_derivate(dx[i])
            else:
                dx[i] = scalings[name].unscale_derivate(dx[i])

    for i, name in enumerate(y_names):
        if name in scaled_names:
            if direction == "scale":
                y[i] = scalings[name].scale(y[i])
            else:
                y[i] = scalings[name].unscale(y[i])

    all_names = x_names + u_names
    all_vars = casadi.vertcat(model.states, model.inputs)

    no_scaling_vars = set(all_names) - set(scaled_names)

    if len(no_scaling_vars) > 0:
        print(
            f"no scaling for the following signals given:\n    {', '.join(sorted(no_scaling_vars))}"
        )

    for i, name in enumerate(all_names):
        var = all_vars[i]

        scaling = scalings[name]

        if direction == "scale":
            expr = scaling.unscale(var)
        else:
            expr = scaling.scale(var)

        dx = casadi.substitute(dx, var, expr)
        y = casadi.substitute(y, var, expr)

    dx = casadi.simplify(dx)
    y = casadi.simplify(y)

    new_states = casadi.MX(model.states)
    new_inputs = casadi.MX(model.inputs)

    new_vars = casadi.vertcat(new_states, new_inputs)

    dx = casadi.substitute(dx, all_vars, new_vars)
    y = casadi.substitute(y, all_vars, new_vars)

    return OdeModel(
        new_states,
        new_inputs,
        dx,
        y,
        y_names=y_names,
        scalings=model.scalings.copy(),
        scaled=(direction == "scale"),
    )


def get_linearized_matrices(
    model: OdeModel,
    x: np.ndarray | casadi.MX | None = None,
    u: np.ndarray | casadi.MX | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[casadi.MX, casadi.MX, casadi.MX, casadi.MX]
    | tuple[casadi.Function, casadi.Function, casadi.Function, casadi.Function]
):

    df_dx = casadi.jacobian(model.dx, model.states)
    df_dx = casadi.Function(
        "df_dx", [model.states, model.inputs], [df_dx], ["x", "u"], ["df_dx"]
    )

    df_du = casadi.jacobian(model.dx, model.inputs)
    df_du = casadi.Function(
        "df_dx", [model.states, model.inputs], [df_du], ["x", "u"], ["df_du"]
    )

    dh_dx = casadi.jacobian(model.y, model.states)
    dh_dx = casadi.Function(
        "dh_dx", [model.states, model.inputs], [dh_dx], ["x", "u"], ["dh_dx"]
    )

    dh_du = casadi.jacobian(model.y, model.inputs)
    dh_du = casadi.Function(
        "dh_du", [model.states, model.inputs], [dh_du], ["x", "u"], ["dh_du"]
    )

    if x is None:
        return (df_dx, df_du, dh_dx, dh_du)
    elif isinstance(x, casadi.MX):
        return (df_dx(x, u), df_du(x, u), dh_dx(x, u), dh_du(x, u))
    else:
        return (
            df_dx(x, u).full(),
            df_du(x, u).full(),
            dh_dx(x, u).full(),
            dh_du(x, u).full(),
        )
