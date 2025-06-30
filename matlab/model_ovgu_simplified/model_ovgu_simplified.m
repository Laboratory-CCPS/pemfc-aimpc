function [dx, y] = model_ovgu_simplified(x, u, p)

    state_vars = {'c_H2_a', 'c_H2O_a', 'c_O2_c', 'c_H2O_c', 'c_N2_c'};
    input_vars = {'n_dot_H2_a_in', 'n_dot_H2O_a_in', ...
                    'n_dot_O2_c_in', 'n_dot_H2O_c_in', 'n_dot_N2_c_in', ...
                    'T_a_in', 'T_c_in', ...
                    'T_cool', ...
                    'p_a_out', 'p_c_out', ...
                    'I_cell'};

    if nargin == 0
        dx.states = state_vars;
        dx.inputs = input_vars;

        y = [];
        return;
    end    

    assert(isempty(setdiff(fieldnames(x), state_vars)));    
    assert(isempty(setdiff(fieldnames(u), input_vars)));
    
    % -------------------------------------------------------------------------------------
    % additional variables
    
    % anode gas temperature (simplified!)
    T_a = u.T_cool;
    % anode pressure
    p_a = p.R * T_a .* (x.c_H2_a + x.c_H2O_a);
    
    % cathode gas temperature (simplified!)
    T_c = u.T_cool;
    % cathode pressure
    p_c = p.R * T_c .* (x.c_O2_c + x.c_H2O_c + x.c_N2_c);
    
    % solid temperature (simplified!)
    T_s = u.T_cool;
    
    % mole fractions of H2O in the anode channel
    xi_H2O_a = x.c_H2O_a ./ (x.c_H2_a + x.c_H2O_a);
    
    % mole fractions of H2O in the cathode channel
    xi_H2O_c = x.c_H2O_c ./ (x.c_O2_c + x.c_H2O_c + x.c_N2_c);
    
    % current density (instead of integral charge balance)
    i_m = u.I_cell / (p.L_x * p.L_z);
     
    % --------------------------------------------------------------------------------------   
    % membrane (only explicit algebraic equations)
    
    % relative humidities on the sides of membrane
    a_H2O_ca = xi_H2O_a .* p_a ./ p.p_sat(T_s);
    a_H2O_cc = xi_H2O_c .* p_c ./ p.p_sat(T_s); 
    
    % correction for more simulation robustness in invalid model region
    % (model invalid fo condensing water vapor)
    % -> changes model behavior for a_H2O_a/c>0.95
    scaling = 300; % scaling parameter smoothness vs. model error 300
    a_H2O_ca = -smoothmax(-a_H2O_ca,-0.97,scaling); %!!!
    a_H2O_cc = -smoothmax(-a_H2O_cc,-0.97,scaling); %!!!
    
    % water content on the anodic side of the membrane
    lambda_am = p.lambda_m_x(a_H2O_ca, T_s);
    
    % water content on the cathodic side of the membrane
    lambda_cm = p.lambda_m_x(a_H2O_cc, T_s);
    
    % membrane water content (simplified!)
    lambda_m = 0.5* (lambda_am+lambda_cm);
    
    % membrane potential (simplified!)
    delta_Phi_m = p.delta_m ./ p.kappa(lambda_m, T_s) .* i_m;
    
    % gradients of the chemical potentials
    % help variables
    xi_H2O_m  = p.xi_H2O(lambda_m);
    xi_H2O_am = p.xi_H2O(lambda_am);
    xi_H2O_cm = p.xi_H2O(lambda_cm);
    xi_H_am   = p.xi_H(lambda_am);
    xi_H_cm   = p.xi_H(lambda_cm);
    % anodic for water
    grad_mu_H2O_am = (p.R * T_s)./(xi_H2O_m) .* (xi_H2O_m - xi_H2O_am)/(0.5*p.delta_m);
    % cathodic for water
    grad_mu_H2O_cm = (p.R * T_s)./(xi_H2O_m) .* (xi_H2O_m - xi_H2O_cm)/(0.5*p.delta_m);
    % anodic for H+
    grad_mu_H_am = (p.R * T_s)./(0.5*xi_H_cm + 0.5*xi_H_am) .* (xi_H_cm - xi_H_am)/(p.delta_m) + p.F * (-delta_Phi_m/p.delta_m);
    % cathodic for H+
    grad_mu_H_cm = (p.R * T_s)./(0.5*xi_H_cm + 0.5*xi_H_am) .* (xi_H_am - xi_H_cm)/(p.delta_m) + p.F * (delta_Phi_m/p.delta_m); % prev. -delta_Phi_m
    
    % concentration of water inside the membrane
    c_H2O_m = lambda_m .* p.rho_m(lambda_m) .* p.x_m(lambda_m);
    
    % water fluxes through the membrane
    % anodic flux
    u.n_dot_H2O_am =  - (p.t_w(lambda_m).*p.kappa(lambda_m, T_s)) / (p.F^2) .* grad_mu_H_am - ...
                    (p.D_w(lambda_m, T_s).*c_H2O_m) ./ (p.R * T_s) .* grad_mu_H2O_am;
    
    % cathodic flux
    u.n_dot_H2O_cm = - (p.t_w(lambda_m).*p.kappa(lambda_m, T_s)) / (p.F^2) .* grad_mu_H_cm - ...
                    (p.D_w(lambda_m, T_s).*c_H2O_m) ./ (p.R * T_s) .* grad_mu_H2O_cm;
    
    % --------------------------------------------------------------------------------------   
    % catalyst layers
    
    % anode catalyst layer mass balances
    r_a = 1/(2*p.F)*i_m;
    u.n_dot_H2_a = r_a;
    u.n_dot_H2O_a = u.n_dot_H2O_am;
    
    % cathode catalyst layer mass balances
    r_c = 1/(2*p.F)*i_m;
    u.n_dot_O2_c = 1/2 * r_c;
    u.n_dot_H2O_c = u.n_dot_H2O_cm - r_c;
    u.n_dot_N2_c = 0;
    
    
    % --------------------------------------------------------------------------------------   
    % GDLs (neglected)
    
    % --------------------------------------------------------------------------------------   
    % channels
    
    % anode channel
    % flow velocity
    v_a = -2*p.K_a/p.delta_z*(u.p_a_out - p_a);
    
    % time derivative of H2 concentration 
    dx.c_H2_a = - 1/p.delta_z .* (v_a .* x.c_H2_a - u.n_dot_H2_a_in) ...
               -  u.n_dot_H2_a /(p.delta_a);
    
    % time derivative of H2O concentration
    dx.c_H2O_a = - 1/p.delta_z .* (v_a .* x.c_H2O_a - u.n_dot_H2O_a_in) ...
                - u.n_dot_H2O_a /(p.delta_a);
    
    % cathode channel
    % flow velocity
    v_c = -2*p.K_c/p.delta_z*(u.p_c_out - p_c);
    
    % time derivative of O2 concentration
    dx.c_O2_c = - 1/p.delta_z .* (v_c .* x.c_O2_c - u.n_dot_O2_c_in) ...
               -  u.n_dot_O2_c /(p.delta_c);
           
    % time derivative of H2O concentration
    dx.c_H2O_c = - 1/p.delta_z .* (v_c .* x.c_H2O_c - u.n_dot_H2O_c_in) ...
               -  u.n_dot_H2O_c /(p.delta_c);
           
    % time derivative of N2 concentration 
    dx.c_N2_c = - 1/p.delta_z .* (v_c .* x.c_N2_c - u.n_dot_N2_c_in);


    % outputs

    % anode gas temperature (simplified!)
    T_a = u.T_cool;
    % anode pressure
    p_a = p.R * T_a .* (x.c_H2_a + x.c_H2O_a);
    
    % cathode gas temperature (simplified!)
    T_c = u.T_cool;
    % cathode pressure
    p_c = p.R * T_c .* (x.c_O2_c + x.c_H2O_c + x.c_N2_c);
    
    % solid temperature (simplified!)
    T_s = u.T_cool;
    
    % mole fractions of H2O in the anode channel
    xi_H2O_a = x.c_H2O_a ./ (x.c_H2_a + x.c_H2O_a);
    
    % mole fractions in the cathode channel
    xi_O2_c = x.c_O2_c ./ (x.c_O2_c + x.c_H2O_c + x.c_N2_c);
    xi_H2O_c = x.c_H2O_c ./ (x.c_O2_c + x.c_H2O_c + x.c_N2_c);
    
    % current density (instead of integral charge balance)
    i_m = u.I_cell / (p.L_x * p.L_z);
    
    % relative humidities on the sides of membrane
    a_H2O_ca = xi_H2O_a .* p_a ./ p.p_sat(T_s);
    a_H2O_cc = xi_H2O_c .* p_c ./ p.p_sat(T_s); 
    
    % % correction for more simulation robustness in invalid model region
    % % (model invalid fo condensing water vapor)
    % % -> changes model behavior for a_H2O_a/c>0.95
    scaling = 300; % scaling parameter smoothness vs. model error 300
    a_H2O_ca = -smoothmax(-a_H2O_ca,-0.97,scaling); %!!!
    a_H2O_cc = -smoothmax(-a_H2O_cc,-0.97,scaling); %!!!
    
    % water content on the anodic side of the membrane
    lambda_am = p.lambda_m_x(a_H2O_ca, T_s);
    
    % water content on the cathodic side of the membrane
    lambda_cm = p.lambda_m_x(a_H2O_cc, T_s);
    
    % membrane water content
    lambda_m = 0.5* (lambda_am+lambda_cm);
    
    % voltage (simplified)
    % cathode activation losses
    % inverse equation to find delta_Phi_c (valid only for p.alpha_c=0.5!)
    delta_Phi_c = -asinh( 1/(2*p.F)*i_m ./(2*p.f_v * p.i_0_ref_c) .* 2*p.F...
        ./ (p_c .* xi_O2_c./p.p_O2_ref) ) ./ (p.alpha_c*2*p.F) .* (p.R*T_s)...
        + p.delta_Phi_c_ref;
    
    % membrane potential
    delta_Phi_m = p.delta_m ./ p.kappa(lambda_m, T_s) .* i_m;
    
    % cell voltage
    y.U_S = delta_Phi_c - delta_Phi_m; % in V for 1 cell (not stack)
    
    % counter-flow case
    y.T_So_A = T_a - 273.15; % from K to °C
    y.p_Si_A = p_a / 1e5; % from Pa to bar
    
    
    % temperature and pressure in the cathode channel
    y.T_So_C = T_c - 273.15; % from K to °C
    y.p_Si_C = p_c / 1e5; % from Pa to bar    
end
