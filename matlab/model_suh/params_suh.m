function p = get_param_suh()

    % Nature
    p.R = 8.3145; %J/mol/K; Universal gas constant
    p.C_p = 1004; %J/Kg/K; Constant pressure Specific heat of air
    p.gamma = 1.4; %1; Ratio of specific heat of air
    p.M_O2 = 32e-3; %kg/mol; Oxygen molar mass
    p.M_N2 = 28e-3; %kg/mol; Nitrogen molar mass
    p.M_v = 18.02e-3; %kg/mol; Vapour molar mass
    p.F = 96485; %C/mol; Faraday constant
    
    % Compressor
    p.eta_cp = 0.7; %1; compressor efficiency
    p.r_c = 0.2286/2; %m; Compressor blade radius
    p.eta_cm = 0.98; %1; motor mechanical Efficiency
    p.k_t = 0.0153; %Nm/A; Motor constant (Pukrushpan, Control of Fuel Cell Power Sytems, page 21)
    p.k_v = 0.0153; %V/(rad/s); Motor constant (Pukrushpan, Control of Fuel Cell Power Sytems, page 21)
    p.R_cm = 0.82; %Ohm; Motor resistance (Pukrushpan, Control of Fuel Cell Power Sytems, page 21)
    p.J_cp = 5e-5; %kg*m^2; compressor and motor inertia
    
    % FC stack
    p.n = 381; %1; Number of cells in fuel-cell stack
    p.T_st = celsius2kelvin(80); %K; Stack temperature
    p.p_sat = satPressure(p.T_st);
    
    % Piping
    p.V_sm = 0.02; %m^3; Supply manifold volume
    p.V_ca = 0.01; %m^3; Cathode volume
    p.k_cain = 0.3629e-5; %kg/s/Pa; Cathode inlet orifice constant
    p.C_D = 0.0124; %1; Cathode outlet throttle discharge coefficient
    p.A_T = 0.002; %m^2; Cathode outlet throttle area
    
    % Inlet/atmospheric air
    p.p_atm = 101325; %Pa; Atmospheric pressure
    p.T_atm = celsius2kelvin(25); %K; Atmospheric temperature
    p.Phi_atm = 0.5; %1; Average ambient air relative humidity
    p.y_O2atm = 0.21; %1; Oxygen mole fraction

    % Assumption: Dry air only consists of O2 and N2
    p.y_N2atm = 1 - p.y_O2atm;
    p.M_aatm  = p.y_O2atm * p.M_O2 + p.y_N2atm * p.M_N2; %Molar Mass atmospheric air
    p.x_O2atm = p.y_O2atm * p.M_O2 / p.M_aatm; %Mass fraction of O2 in atmospheric air
    p.x_N2atm = p.y_N2atm * p.M_N2 / p.M_aatm; %Mass fraction of N2 in atmospheric air
    p.p_vatm = p.Phi_atm * p.p_sat; %Pa; Vapor pressure in atm. air
    p.w_atm = p.M_v / p.M_aatm * p.p_vatm / (p.p_atm - p.p_vatm); %kg/kg; Vapor content in atm. air
    p.R_aatm = p.R / p.M_aatm; %J/(kg*K) Air specific gas constant
    p.rho_aatm = p.p_atm / (p.R_aatm * p.T_atm); %kg/m^3; air density


end

