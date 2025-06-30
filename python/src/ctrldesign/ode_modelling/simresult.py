from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

import numpy as np
from matplotlib import pyplot as plt

from .casadi_helper import casadi_vars_to_str
from .discrete_model import DiscreteModel
from .helper import find_element_idx
from .model_helper import scale_data, unscale_data
from .ode_model import OdeModel
from .scaling import Scaling
from .signalinfo import SignalInfo


@dataclass
class SimResult:
    t: np.ndarray
    u: np.ndarray
    x: np.ndarray
    y: np.ndarray
    y_ft: Optional[np.ndarray]
    u_names: tuple[str, ...]
    x_names: tuple[str, ...]
    y_names: tuple[str, ...]

    scalings: dict[str, Scaling]
    scaled: bool

    constraints: dict[str, tuple[float, float]]

    discrete_model: bool
    Ts: float
    desc: str

    @staticmethod
    def from_data(
        model: OdeModel | DiscreteModel,
        t: np.ndarray,
        u: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        y_ft: Optional[np.ndarray] = None,
    ):
        assert (len(t.shape) == 1) or (t.shape[1] == 1)

        nt = len(t)
        nu = model.inputs.numel()
        nx = model.states.numel()
        ny = len(model.y_names)

        assert u.shape == (nu, nt)
        assert x.shape == (nx, nt)
        assert y.shape == (ny, nt)
        assert (y_ft is None) or (y_ft.shape == (ny, nt))

        discrete_model = isinstance(model, DiscreteModel)

        return SimResult(
            t=t,
            u=u,
            x=x,
            y=y,
            y_ft=y_ft,
            u_names=casadi_vars_to_str(model.inputs),
            x_names=casadi_vars_to_str(model.states),
            y_names=model.y_names,
            scalings=model.scalings,
            scaled=model.scaled,
            constraints={},
            discrete_model=discrete_model,
            Ts=model.Ts if discrete_model else 0.0,
            desc="",
        )

    def copy(self):
        return SimResult(
            t=self.t.copy(),
            u=self.u.copy(),
            x=self.x.copy(),
            y=self.y.copy(),
            y_ft=None if self.y_ft is None else self.y_ft.copy(),
            u_names=self.u_names,
            x_names=self.x_names,
            y_names=self.y_names,
            scalings=self.scalings.copy(),
            scaled=self.scaled,
            constraints=self.constraints.copy(),
            discrete_model=self.discrete_model,
            Ts=self.Ts,
            desc=self.desc,
        )

    def scale(self):
        if self.scaled:
            print("data already scaled")
            return

        self.u = scale_data(self.scalings, self.u_names, self.u)
        self.x = scale_data(self.scalings, self.x_names, self.x)
        self.y = scale_data(self.scalings, self.y_names, self.y)

        if self.y_ft is not None:
            self.y_ft = scale_data(self.scalings, self.y_names, self.y_ft)

        self.scaled = True

    def unscale(self):
        if not self.scaled:
            print("data already unscaled")
            return

        self.u = unscale_data(self.scalings, self.u_names, self.u)
        self.x = unscale_data(self.scalings, self.x_names, self.x)
        self.y = unscale_data(self.scalings, self.y_names, self.y)

        if self.y_ft is not None:
            self.y_ft = unscale_data(self.scalings, self.y_names, self.y_ft)

        self.scaled = False

    def _get_signal_values(
        self,
        name: str,
        x: Optional[np.ndarray],
        u: Optional[np.ndarray],
        y: Optional[np.ndarray],
    ) -> float | np.ndarray:
        if (idx := find_element_idx(self.x_names, name)) is not None:
            if x is None:
                raise ValueError(f"'{name}' is a state, but no state values provided")

            if len(x.shape) == 1:
                return x[idx]
            else:
                return x[idx, :]

        if (idx := find_element_idx(self.u_names, name)) is not None:
            if u is None:
                raise ValueError(f"'{name}' is an input, but no input values provided")

            if len(u.shape) == 1:
                return u[idx]
            else:
                return u[idx, :]

        if (idx := find_element_idx(self.y_names, name)) is not None:
            if y is None:
                raise ValueError(
                    f"'{name}' is an output, but no output values provided"
                )

            if len(y.shape) == 1:
                return y[idx]
            else:
                return y[idx, :]

        raise ValueError(f"'{name}' is not a signal")

    def get_signal_values(
        self,
        names: str | Sequence[str],
        return_tuple: bool = False,
    ) -> float | np.ndarray | tuple[float, ...] | tuple[np.ndarray]:

        if isinstance(names, str):
            names = [names]

        values: list[np.ndarray | float] = []

        for name in names:
            values.append(self._get_signal_values(name, self.x, self.u, self.y))

        if return_tuple:
            return tuple(values)

        else:
            return np.vstack(values)


@dataclass
class AddSignal:
    values: np.ndarray
    t: Optional[np.ndarray] = None
    label: Optional[str] = None
    on_top: bool = True
    step_plot: bool = False
    format_kwargs: Optional[dict[str, Any]] = None


_BASECOLORS: tuple[tuple[float, float, float], ...] = [
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
]


def plot_sim_results(
    results: SimResult | Sequence[SimResult],
    plot_signals: Optional[Sequence[str]] = None,
    *,
    add_signals: Optional[dict[str, AddSignal | Sequence[AddSignal]]] = None,
    reuse_figures: bool = False,
    signal_infos: Optional[dict[str, SignalInfo]] = None,
    auto_unscale: bool = True,
):
    if isinstance(results, SimResult):
        results = (results,)

    if plot_signals is None:
        plot_signals = ["@states, @outputs, @inputs"]
    elif isinstance(plot_signals, str):
        plot_signals = [plot_signals]

    if add_signals is None:
        add_signals = {}

    n_results = len(results)

    x_names: list[str] = []
    y_names: list[str] = []
    u_names: list[str] = []

    scaled = results[0].scaled

    wc_results: list[SimResult] = []

    for i in range(len(results)):
        res = results[i]
        for x in res.x_names:
            if x not in x_names:
                x_names.append(x)

        for y in res.y_names:
            if y not in y_names:
                y_names.append(y)

        for u in res.u_names:
            if u not in u_names:
                u_names.append(u)

        if res.scaled:
            if auto_unscale:
                res = res.copy()
                res.unscale()
            elif res.scaled != scaled:
                raise ValueError("the results must be all scaled or all unscaled")

        wc_results.append(res)

    results = wc_results

    if auto_unscale:
        scaled = False

    labels = []

    if n_results > 1 or results[0].desc != "":
        for i, res in enumerate(results):
            if res.desc == "":
                labels.append(f"sim. {i}")
            else:
                labels.append(res.desc)

    ax_link_master: Optional[plt.Axes] = None

    for i_fig in range(len(plot_signals)):
        (title_str, signals) = parse_plot_info(plot_signals[i_fig])

        signals = replace_category_names(signals, x_names, u_names, y_names)

        if title_str == "":
            if reuse_figures:
                plt.clf()
            else:
                plt.figure()
        elif reuse_figures:
            plt.figure(title_str)
            plt.clf()
        else:
            plt.figure(title_str)
            plt.clf()

        n_subplots = len(signals)

        n_cols = 1 if n_subplots < 5 else 2

        n_rows = int(np.ceil(n_subplots / n_cols))

        vh: list[plt.Axes] = []

        for ip, signal in enumerate(signals):

            vh.append(plt.subplot(n_rows, n_cols, ip + 1, sharex=ax_link_master))

            if ax_link_master is None:
                ax_link_master = vh[-1]

            has_labels = False

            # Plot all the additional signals to the current signal that are plotted in
            # the background
            if signal in add_signals:
                addsigs = add_signals[signal]

                if isinstance(addsigs, AddSignal):
                    addsigs = [addsigs]

                for addsig in addsigs:
                    if not addsig.on_top:
                        has_labels |= _plot_add_signal(
                            vh[-1], addsig, res.t, signal, signal_infos, scaled
                        )

            for ir, res in enumerate(results):
                if ir >= len(_BASECOLORS):
                    raise ValueError(
                        f"more than {len(_BASECOLORS)} cannot be plotted together"
                    )

                basecolor = _BASECOLORS[ir]

                label = labels[ir] if len(labels) > ir else None

                has_labels |= plot_(
                    vh[-1],
                    res,
                    signal,
                    signal_infos,
                    basecolor,
                    scaled,
                    label,
                )

            # Plot all the additional signals to the current signal that are plotted on
            # top
            if signal in add_signals:
                addsigs = add_signals[signal]

                if isinstance(addsigs, AddSignal):
                    addsigs = [addsigs]

                for addsig in addsigs:
                    if addsig.on_top:
                        has_labels |= _plot_add_signal(
                            vh[-1], addsig, res.t, signal, signal_infos, scaled
                        )

            if has_labels:
                plt.legend()

            if ip >= n_subplots - n_cols:
                if "time" in signal_infos:
                    info = signal_infos["time"]
                    vh[-1].set_xlabel(f"{info.tex} / {info.disp_unit}")
                else:
                    vh[-1].set_xlabel("time")

            if (signal_infos is not None) and (signal in signal_infos):
                info = signal_infos[signal]

                sig_title = info.tex

                if scaled:
                    sig_ylabel = f"{info.tex} (scaled)"
                elif info.disp_unit == "":
                    sig_ylabel = info.tex
                else:
                    sig_ylabel = f"{info.tex} / {info.disp_unit}"

            else:
                sig_title = signal

                if scaled:
                    sig_ylabel = f"{signal} (scaled)"
                else:
                    sig_ylabel = signal

            vh[-1].set_title(sig_title)
            vh[-1].set_ylabel(sig_ylabel)

            vh[-1].grid(True)

        plt.tight_layout()


def plot_(
    ax: plt.Axes,
    res: SimResult,
    signal: str,
    signal_infos: dict[str, SignalInfo],
    basecolor: tuple[float, float, float],
    scaled: bool,
    label: Optional[str],
) -> bool:
    stair_plot = False

    (kind, idx) = find_signal(res, signal)

    if kind == "x":
        values = res.x[idx, :]
        t = res.t
    elif kind == "y":
        values = res.y[idx, :]
        t = res.t

        if res.y_ft is not None:
            values_ft = res.y_ft[idx, :]

            t = np.hstack((t.reshape((-1, 1)), t.reshape((-1, 1)))).reshape((-1,))[1:-1]
            values = np.hstack(
                (values.reshape((-1, 1)), values_ft.reshape((-1, 1)))
            ).reshape((-1,))[1:-1]

    elif kind == "u":
        stair_plot = True
        values = res.u[idx, :]
        t = res.t
    else:
        return False

    if (not scaled) and (signal_infos is not None) and (signal in signal_infos):
        info = signal_infos[signal]
    else:
        info = SignalInfo(signal, "", "", lambda x: x)

    if (signal_infos is not None) and ("time" in signal_infos):
        t_info = signal_infos["time"]
    else:
        t_info = SignalInfo("time", "", "", lambda x: x)

    if not stair_plot:
        ax.plot(t_info.disp_fct(t), info.disp_fct(values), color=basecolor, label=label)
    else:
        ax.step(
            t_info.disp_fct(t),
            info.disp_fct(values),
            color=basecolor,
            where="post",
            label=label,
        )

    if signal in res.constraints:
        (min, max) = res.constraints[signal]

        if min > -np.inf:
            ax.axhline(
                info.disp_fct(min),
                linestyle=":",
                color="k",
                label=None if label is None else f"_{label}_lower_constraint",
            )

        if max < np.inf:
            ax.axhline(
                info.disp_fct(max),
                linestyle=":",
                color="k",
                label=None if label is None else f"_{label}_upper_constraint",
            )

    return label is not None


def _plot_add_signal(
    ax: plt.Axes,
    addsig: AddSignal,
    res_t: np.ndarray,
    signal: str,
    signal_infos: Optional[dict[str, SignalInfo]],
    scaled: bool,
) -> bool:
    if (not scaled) and (signal_infos is not None) and (signal in signal_infos):
        info = signal_infos[signal]
    else:
        info = SignalInfo(signal, "", "", lambda x: x)

    if (signal_infos is not None) and ("time" in signal_infos):
        t_info = signal_infos["time"]
    else:
        t_info = SignalInfo("time", "", "", lambda x: x)

    if addsig.t is None:
        t = res_t
    else:
        t = addsig.t

    values = addsig.values.reshape((-1,))

    if addsig.label is None:
        label = "_none"
    else:
        label = addsig.label

    if addsig.format_kwargs is None:
        format = {}
    else:
        format = addsig.format_kwargs

    if addsig.step_plot:
        ax.step(
            t_info.disp_fct(t),
            info.disp_fct(values),
            where="post",
            label=label,
            **format,
        )
    else:
        ax.plot(
            t_info.disp_fct(t),
            info.disp_fct(values),
            label=label,
            **format,
        )

    return addsig.label is not None


def parse_plot_info(s: str) -> tuple[str, list[str]]:
    title_signals = s.split(":")

    if len(title_signals) == 1:
        title_signals = [""] + title_signals

    title = title_signals[0].strip()
    signals = [sig.strip() for sig in title_signals[1].split(",")]

    return (title, signals)


def replace_category_names(
    names: Sequence[str],
    x_names: Iterable[str],
    u_names: Iterable[str],
    y_names: Iterable[str],
) -> list[str]:
    new_names = []

    for name in names:
        if name == "@states":
            new_names.extend(x_names)
        elif name == "@inputs":
            new_names.extend(u_names)
        elif name == "@outputs":
            new_names.extend(y_names)
        elif name == "@all":
            new_names.extend(x_names)
            new_names.extend(y_names)
            new_names.extend(u_names)
        else:
            new_names.append(name)

    return new_names


def find_index(el: Any, it: Iterable[Any]) -> int:
    for i, v in enumerate(it):
        if v == el:
            return i

    else:
        return -1


def find_signal(result: SimResult, name: str) -> tuple[Literal["x", "u", "y", ""], int]:
    idx = find_index(name, result.x_names)
    if idx >= 0:
        return ("x", idx)

    idx = find_index(name, result.u_names)
    if idx >= 0:
        return ("u", idx)

    idx = find_index(name, result.y_names)
    if idx >= 0:
        return ("y", idx)

    return ("", -1)
