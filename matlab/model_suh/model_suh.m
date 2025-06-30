function [dx, y] = model_suh(x, u, p)

    state_vars = {'p_O2', 'p_N2', 'w_cp', 'p_sm'};
    input_vars = {'v_cm', 'I_st'};

    if nargin == 0
        dx.states = state_vars;
        dx.inputs = input_vars;

        y.p_O2 = Scaling.FromRange(0.1e5, 0.4e5);
        y.p_N2 = Scaling.FromRange(0.5e5, 3e5);
        y.w_cp = Scaling.FromRange(rpm2rad(0), rpm2rad(105e3));
        y.p_sm = Scaling.FromRange(0.5e5, 4e5);
        y.v_cm = Scaling.FromRange(50, 500);
        y.I_st = Scaling.FromRange(0, 350);
        y.lambda_O2 = Scaling.FromRange(1, 3);
        return;
    end

    assert(isempty(setdiff(fieldnames(x), state_vars)));
    assert(isempty(setdiff(fieldnames(u), input_vars)));

    p_O2 = x.p_O2;
    p_N2 = x.p_N2;
    w_cp = x.w_cp;
    p_sm = x.p_sm;

    v_cm = u.v_cm;
    I_st = u.I_st;

    %cathode pressure
    p_ca = p_O2 + p_N2 + p.p_sat;

    % cathode inlet flow
    W_cain = p.k_cain * (p_sm - p_ca);
    W_O2in = p.x_O2atm / (1 + p.w_atm) * W_cain;
    W_N2in = p.x_N2atm / (1 + p.w_atm) * W_cain;

    %rate of O2 consumption
    W_O2rct = p.M_O2 * p.n * I_st / (4 * p.F);

    %cathode output flow
    W_caOut = get_W_caOut(p_ca, p);

    h1 = p.M_O2 * p_O2 + p.M_N2 * p_N2 + p.M_v * p.p_sat;
    W_O2out = W_caOut * (p.M_O2 * p_O2) / h1;
    W_N2out = W_caOut * (p.M_N2 * p_N2) / h1;

    %compressor (flow, output temperature, torque)
    [W_cp, T_cp, t_cp] = compressor(p.p_atm, p.T_atm, w_cp, p_sm, p);

    %compressor Motor torque
    t_cm = p.eta_cm * p.k_t * (v_cm - p.k_v * w_cp) / p.R_cm;

    % Oxyygen excess ratio
    lambda_O2 = W_O2in / W_O2rct;

    dx.p_O2 = p.R * p.T_st / (p.M_O2 * p.V_ca) * (W_O2in - W_O2out - W_O2rct);
    dx.p_N2 = p.R * p.T_st / (p.M_N2 * p.V_ca) * (W_N2in - W_N2out);
    dx.w_cp = (t_cm - t_cp) / p.J_cp;
    dx.p_sm = p.R * T_cp / (p.M_aatm * p.V_sm) * (W_cp - W_cain);

    y.lambda_O2 = lambda_O2;
end

function W_caOut = get_W_caOut(p_ca, p)
    % Calculates cathode exit flow from cathode pressure.
    % Assumes that this flow exits into atmospheric air which has
    % Pressure defined in FCmodel_Suh.c.p_atm
    %
    % Usage:
    %   Input:
    %       p_ca    cathode pressure
    %   Output:
    %       W       cathode exit flow
    [p_interp, W_interp] = get_W_caOut_lookup_table(p);

    if isnumeric(p_ca)
        W_caOut = interp1(p_interp, W_interp, p_ca);
    else
        W_caOut_interp = casadi.interpolant('W_caOut', 'linear', {p_interp}, W_interp);
        W_caOut = W_caOut_interp(p_ca);
    end
end
