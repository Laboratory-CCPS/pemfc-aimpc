import dataclasses
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import casadi
import numpy as np

from src.ctrldesign.ode_modelling.scaling import Scaling
from src.ctrldesign.ode_modelling.signalinfo import SignalInfo

from .conversions import bar2pa, celsius2kelvin, rad2rpm, rpm2rad
from .vapor_properties import saturation_pressure


@dataclass
class ParamsSuh:
    R: float  # %J/mol/K; Universal gas constant
    C_p: float  # %J/Kg/K; Constant pressure Specific heat of air
    gamma: float  # %1; Ratio of specific heat of air
    M_O2: float  # %kg/mol; Oxygen molar mass
    M_N2: float  # %kg/mol; Nitrogen molar mass
    M_v: float  # %kg/mol; Vapour molar mass
    F: float  # %C/mol; Faraday constant

    # Compressor
    eta_cp: float  # %1; compressor efficiency
    r_c: float  # %m; Compressor blade radius
    eta_cm: float  # %1; motor mechanical Efficiency
    k_t: float  # %Nm/A; Motor constant (Pukrushpan, Control of Fuel Cell Power Sytems, page 21)
    k_v: float  # %V/(rad/s); Motor constant (Pukrushpan, Control of Fuel Cell Power Sytems, page 21)
    R_cm: float  # %Ohm; Motor resistance (Pukrushpan, Control of Fuel Cell Power Sytems, page 21)
    J_cp: float  # %kg*m^2; compressor and motor inertia

    # FC stack
    n: float  # %1; Number of cells in fuel-cell stack
    T_st: float  # %K; Stack temperature
    p_sat: float  #

    # Piping
    V_sm: float  # %m^3; Supply manifold volume
    V_ca: float  # %m^3; Cathode volume
    k_cain: float  # %kg/s/Pa; Cathode inlet orifice constant
    C_D: float  # %1; Cathode outlet throttle discharge coefficient
    A_T: float  # %m^2; Cathode outlet throttle area

    # Inlet/atmospheric air
    p_atm: float  # %Pa; Atmospheric pressure
    T_atm: float  # %K; Atmospheric temperature
    Phi_atm: float  # %1; Average ambient air relative humidity
    y_O2atm: float  # %1; Oxygen mole fraction

    # Assumption: Dry air only consists of O2 and N2
    y_N2atm: float  #
    M_aatm: float  # %Molar Mass atmospheric air
    x_O2atm: float  # %Mass fraction of O2 in atmospheric air
    x_N2atm: float  # %Mass fraction of N2 in atmospheric air
    p_vatm: float  # %Pa; Vapor pressure in atm. air
    w_atm: float  # %kg/kg; Vapor content in atm. air
    R_aatm: float  # %J/(kg*K) Air specific gas constant
    rho_aatm: float  # %kg/m^3; air density

    def copy(self):
        return dataclasses.replace(self)


def params() -> ParamsSuh:
    R = 8.3145
    M_O2 = 32e-3
    M_N2 = 28e-3
    M_v = 18.02e-3

    T_st = celsius2kelvin(80)

    p_sat = saturation_pressure(T_st)
    Phi_atm = 0.5
    p_atm = 101325
    T_atm = celsius2kelvin(25)
    Phi_atm = 0.5
    y_O2atm = 0.21
    y_N2atm = 1 - y_O2atm
    p_vatm = Phi_atm * p_sat

    M_aatm = y_O2atm * M_O2 + y_N2atm * M_N2
    R_aatm = R / M_aatm

    return ParamsSuh(
        # Nature
        R=R,
        C_p=1004,
        gamma=1.4,
        M_O2=M_O2,
        M_N2=M_N2,
        M_v=M_v,
        F=96485,
        # Compressor
        eta_cp=0.7,
        r_c=0.2286 / 2,
        eta_cm=0.98,
        k_t=0.0153,
        k_v=0.0153,
        R_cm=0.82,
        J_cp=5e-5,
        # FC stack
        n=381,
        T_st=T_st,
        p_sat=p_sat,
        # Piping
        V_sm=0.02,
        V_ca=0.01,
        k_cain=0.3629e-5,
        C_D=0.0124,
        A_T=0.002,
        # Inlet/atmospheric air
        p_atm=p_atm,
        T_atm=T_atm,
        Phi_atm=Phi_atm,
        y_O2atm=y_O2atm,
        # Assumption: Dry air only consists of O2 and N2
        y_N2atm=y_N2atm,
        M_aatm=M_aatm,
        x_O2atm=y_O2atm * M_O2 / M_aatm,
        x_N2atm=y_N2atm * M_N2 / M_aatm,
        p_vatm=p_vatm,
        w_atm=M_v / M_aatm * p_vatm / (p_atm - p_vatm),
        R_aatm=R_aatm,
        rho_aatm=p_atm / (R_aatm * T_atm),
    )


def signal_infos() -> dict[str, SignalInfo]:
    """Specifies signal properties for plotting"""
    return {
        "time": SignalInfo("time", "s", "s", lambda x: x),
        "p_O2": SignalInfo("p_O2", "Pa", "bar", lambda x: x * 1e-5),
        "p_N2": SignalInfo("p_N2", "Pa", "bar", lambda x: x * 1e-5),
        "p_sm": SignalInfo("p_sm", "Pa", "bar", lambda x: x * 1e-5),
        "w_cp": SignalInfo("ω_cp", "rad/s", "krpm", lambda x: rad2rpm(x) * 1e-3),
        "lambda_O2": SignalInfo("λ_O2", "1", "", lambda x: x),
        "I_st": SignalInfo("I_st", "A", "A", lambda x: x),
        "v_cm": SignalInfo("v_cm", "V", "V", lambda x: x),
        "v_st": SignalInfo("v_st", "V", "V", lambda x: x),
        "P_st": SignalInfo("P_st", "W", "kW", lambda x: x * 1e-3),
        "P_net": SignalInfo("P_net", "W", "kW", lambda x: x * 1e-3),
        "eta_st": SignalInfo("η_st", "1", "", lambda x: x),
    }


def model(xup: Optional[tuple[dict[str, Any], dict[str, Any], ParamsSuh]] = None):
    _STATE_VARS = ("p_O2", "p_N2", "w_cp", "p_sm")
    _INPUT_VARS = ("v_cm", "I_st")

    if xup is None:
        vars = {
            "states": _STATE_VARS,
            "inputs": _INPUT_VARS,
        }

        scalings = {
            "p_O2": Scaling.from_range(0.1e5, 0.4e5),
            "p_N2": Scaling.from_range(0.5e5, 3e5),
            "w_cp": Scaling.from_range(rpm2rad(0), rpm2rad(105e3)),
            "p_sm": Scaling.from_range(0.5e5, 4e5),
            "v_cm": Scaling.from_range(50, 250),
            "I_st": Scaling.from_range(0, 350),
            "lambda_O2": Scaling.from_range(1, 3),
            "P_net": Scaling.from_range(20e3, 80e3),
            "P_st": Scaling.from_range(20e3, 80e3),
            "eta_st": Scaling.from_range(0.7, 1),
            "v_st": Scaling.from_range(250, 290),
        }

        return (vars, scalings)

    (x, u, p) = xup

    assert all(f in _STATE_VARS for f in x.keys())
    assert all(f in _INPUT_VARS for f in u.keys())

    p_O2 = x["p_O2"]
    p_N2 = x["p_N2"]
    w_cp = x["w_cp"]
    p_sm = x["p_sm"]

    v_cm = u["v_cm"]
    I_st = u["I_st"]

    # cathode pressure
    p_ca = p_O2 + p_N2 + p.p_sat

    # cathode inlet flow
    W_cain = p.k_cain * (p_sm - p_ca)
    W_O2in = p.x_O2atm / (1 + p.w_atm) * W_cain
    W_N2in = p.x_N2atm / (1 + p.w_atm) * W_cain

    # rate of O2 consumption
    W_O2rct = p.M_O2 * p.n * I_st / (4 * p.F)

    # cathode output flow
    W_caOut = _get_W_caOut(p_ca, p)

    h1 = p.M_O2 * p_O2 + p.M_N2 * p_N2 + p.M_v * p.p_sat
    W_O2out = W_caOut * (p.M_O2 * p_O2) / h1
    W_N2out = W_caOut * (p.M_N2 * p_N2) / h1

    # compressor (flow, output temperature, torque)
    [W_cp, T_cp, t_cp] = _compressor(p.p_atm, p.T_atm, w_cp, p_sm, p)

    # compressor Motor torque
    t_cm = p.eta_cm * p.k_t * (v_cm - p.k_v * w_cp) / p.R_cm

    # Oxyygen excess ratio
    lambda_O2 = W_O2in / W_O2rct

    v_st = p.n * _get_cell_voltage(I_st, p_ca, p_O2, p)
    P_st = v_st * I_st
    P_cm = v_cm * (v_cm - p.k_v * w_cp) / p.R_cm
    P_net = P_st - P_cm
    eta_st = P_net / P_st

    dx = OrderedDict()
    dx["p_O2"] = p.R * p.T_st / (p.M_O2 * p.V_ca) * (W_O2in - W_O2out - W_O2rct)
    dx["p_N2"] = p.R * p.T_st / (p.M_N2 * p.V_ca) * (W_N2in - W_N2out)
    dx["w_cp"] = (t_cm - t_cp) / p.J_cp
    dx["p_sm"] = p.R * T_cp / (p.M_aatm * p.V_sm) * (W_cp - W_cain)

    y = OrderedDict()
    y["lambda_O2"] = lambda_O2
    y["v_st"] = v_st
    y["P_st"] = P_st
    y["P_net"] = P_net
    y["eta_st"] = eta_st

    return (dx, y)


def _compressor(p_in, T_in, w, p_out, p: ParamsSuh):

    # Avoid NaNs and Inf in case w = 0
    if not isinstance(w, casadi.MX) and w <= 0:
        W = 0
        T_out = T_in
        torque = 0
        return (W, T_out, torque)

    # (Pukrushpan, Control of Fuel Cell Power Sytems, page 17 f)

    # Constants differing from rest of model
    R_a = 286.9  # %J/(kg*K); air gas constant
    rho_a = 1.23  # %kg/m^3; Air density

    # K; ideal and real Temperature increase by compressor
    T_inc_ideal = T_in * ((p_out / p_in) ** ((p.gamma - 1) / p.gamma) - 1)
    T_inc_real = T_inc_ideal / p.eta_cp

    delta = p_in / 101325  # %1; Pressure relative to 1 atm
    sTheta = np.sqrt(T_in / 288)  # %1; Temperature relative to 288 K
    w_cr = w / sTheta  # ; %rad/s
    U_c = w_cr * p.r_c  # ; %m/s; Blade tip speed
    Psi = p.C_p * T_inc_ideal / (U_c**2 * 0.5)  # ; %1; head parameter (compressor work)

    speed_sound = np.sqrt(p.gamma * R_a * T_in)  # ; %m/s; speed of sound in air
    M = U_c / speed_sound  # ; %1; mach number

    # coefficients defined by (Pukrushpan, Control of Fuel Cell Power Sytems, page 19)
    a4 = -3.69906e-5
    a3 = 2.70399e-4
    a2 = -5.36235e-4
    a1 = -4.63685e-5
    a0 = 2.21195e-3
    b2 = 1.76567
    b1 = -1.34837
    b0 = 2.44419
    c5 = -9.78755e-3
    c4 = 0.10581
    c3 = -0.42937
    c2 = 0.80121
    c1 = -0.68344
    c0 = 0.43331

    Phi_max = a4 * M**4 + a3 * M**3 + a2 * M**2 + a1 * M + a0
    beta = b2 * M**2 + b1 * M + b0
    Psi_max = c5 * M**5 + c4 * M**4 + c3 * M**3 + c2 * M**2 + c1 * M + c0

    # Normalized compressor flow rate
    Phi = Phi_max * (1 - np.exp(beta * (Psi / Psi_max - 1)))

    # Flow
    W_cr = Phi * rho_a * np.pi * p.r_c**2 * U_c
    W = W_cr * delta / sTheta

    # Output temperature
    T_out = T_in + T_inc_real

    # torque
    P = p.C_p * T_inc_real * W
    torque = P / w

    return (W, T_out, torque)


def _get_W_caOut(p_ca, p: ParamsSuh):
    (p_interp, W_interp) = _get_W_caOut_lookup_table(p)

    if not isinstance(p_ca, casadi.MX):
        W_caOut = np.interp(p_interp, W_interp, p_ca)
    else:
        W_caOut_interp = casadi.interpolant(
            "W_caOut", "linear", [[v for v in p_interp]], [v for v in W_interp]
        )
        W_caOut = W_caOut_interp(p_ca)

    return W_caOut


def _get_W_caOut_lookup_table(p: ParamsSuh):
    # From equations 2.15 and 2.16
    c1 = (p.gamma - 1) / p.gamma
    c2 = p.C_D * p.A_T / np.sqrt(p.R * p.T_st)

    # normal flow:
    def W_normal(p_cathode):
        return (
            c2
            * p_cathode
            * (p.p_atm / p_cathode) ** (1 / p.gamma)
            * np.sqrt(2 / c1)
            * np.sqrt(1 - (p.p_atm / p_cathode) ** c1)
        )

    # # choked flow (linear):
    def W_choked(p_cathode):
        return (
            c2
            * p_cathode
            * np.sqrt(p.gamma)
            * np.sqrt((2 / (p.gamma + 1)) ** ((p.gamma + 1) / (p.gamma - 1)))
        )

    # # Critical Pressure
    # # when p_cathode < p_crit, use W_normal
    p_crit = p.p_atm / ((2 / (p.gamma + 1)) ** (1 / c1))

    # # Create interpolation table for W_normal
    p_ca = np.linspace(p.p_atm, p_crit, 100)

    W1 = np.zeros((len(p_ca),))

    for ii in range(len(p_ca)):
        W1[ii] = W_normal(p_ca[ii])

    # # For W_choked, just use last value of W_normal and some other
    # # value sinc W_choked is linear
    p_end = 10 * p.p_atm

    # # Combine segments to interpolation table
    p_interp = np.concat((np.array([0]), p_ca, np.array([p_end])))
    W_interp = np.concat((np.array([0]), W1, np.array([W_choked(p_end)])))

    return (p_interp, W_interp)


def _get_cell_voltage(I_fc, p_ca, p_O2, p: ParamsSuh):

    A_fc_cm2 = 280e-2
    i_fc = I_fc / A_fc_cm2

    T_fc = p.T_st

    p_ca_bar = p_ca * 1e-5
    p_sat_bar = p.p_sat * 1e-5
    p_O2_bar = p_O2 * 1e-5

    v0 = (
        0.279
        - 8.5e-4 * (T_fc - 298.15)
        + 4.3085e-5
        * T_fc
        * (
            np.log((p_ca_bar - p_sat_bar) / 1.01325)
            + 0.5 * np.log(0.1173 * (p_ca_bar - p_sat_bar) / 1.01325)
        )
    )

    va = (
        (-1.618e-5 * T_fc + 1.618e-2) * (p_O2_bar / 0.1173 + p_sat_bar) ** 2
        + (1.8e-4 * T_fc - 0.166) * (p_O2_bar / 0.1173 + p_sat_bar)
        + (-5.8e-4 * T_fc + 0.5736)
    )

    c1 = 10
    c3 = 2
    i_max = 2.2
    b11 = 0.05139
    b12 = 0.00326
    b2 = 350

    c2 = (7.16e-4 * T_fc - 0.622) * (p_O2_bar / 0.1173 + p_sat_bar) + (
        -1.45e-3 * T_fc + 1.68
    )

    v_conc = i_fc * (c2 * i_fc / i_max) ** c3

    lambda_m = 14

    sigma_m = (b11 * lambda_m - b12) * np.exp(b2 * (1 / 303 - 1 / T_fc))
    t_m = 0.01275e-2
    R_ohm = t_m / sigma_m
    v_ohm = I_fc * R_ohm

    v_act = v0 + va * (1 - np.exp(-c1 * i_fc))

    p_H2_bar = p_ca_bar
    E = (
        1.229
        - 8.5e-4 * (T_fc - 298.15)
        + 4.3085e-5 * T_fc * (np.log(p_H2_bar) + 0.5 * np.log(p_O2_bar))
    )

    return E - v_act - v_ohm - 0 * v_conc


def check_state(
    x: np.ndarray[float] | Sequence[float],
    p: ParamsSuh,
    dp_margin: float = bar2pa(0.025),
) -> tuple[bool, Optional[float]]:
    """Check if the given state x is physically plausible

    If the state is non-physical or leads to behaviour not accounted by the modelling
    (as a reversed air flow) it returns (False, None).
    If the state is valid, the compressor voltage $v_cm$ that corresponds to the steady
    state given by x is calculated and returned as second item of the tuple:
    (True, v_cm)

    dp_marging specifies a small offset to ensure that the air flows from the compressor
    into the cathode and then into the ambient air.
    """
    p_O2 = x[0]
    p_N2 = x[1]
    w_cp = x[2]
    p_sm = x[3]

    p_ca = p_O2 + p_N2 + p.p_sat

    if (p_sm < p_ca + dp_margin) or (p_ca < p.p_atm + dp_margin):
        return (False, None)

    [W_cp, _, t_cp] = _compressor(p.p_atm, p.T_atm, w_cp, p_sm, p)

    if W_cp <= 0:
        return (False, None)

    v_cm = t_cp / (p.eta_cm * p.k_t) * p.R_cm + w_cp * p.k_v

    return (True, v_cm)


def get_validation_input(Ts: float) -> tuple[np.ndarray, np.ndarray]:
    t0 = 0
    tend = 30

    t = np.arange(t0, tend, Ts).reshape((1, -1))

    def I_load(t: np.ndarray) -> np.ndarray:
        return (
            100.0
            + 80 * (t >= 2)
            + 40 * (t >= 6)
            - 20 * (t >= 10)
            + 60 * (t >= 14)
            + 60 * (t >= 22)
        )

    def v_cm(t: np.ndarray) -> np.ndarray:
        return (
            100.0
            + 55 * (t >= 2)
            + 25 * (t >= 6)
            - 10 * (t >= 10)
            + 40 * (t >= 14)
            + 25 * (t >= 18)
            + 15 * (t >= 25)
        )

    x0 = np.array([0.1096e5, 0.7502e5, 5.4982e3, 1.4326e5])

    u = np.vstack((v_cm(t), I_load(t)))

    return (u, x0)
