function [W, T_out, torque] = compressor(p_in, T_in, w, p_out, p)
    % compressor model.
    %
    % Usage:
    % [W, T_out, torque] = ...
    %   FCmodel_Suh.compressor(obj, p_in, T_in, w, p_out)
    %
    %   Calculates:
    %       W       compressor flow in kg/s
    %       T_out   output flow temperature in K
    %       torque  reqired torque to drive the compressor in Nm
    %   Input:
    %       p_in    input pressure in Pa
    %       T_in    input temperature in K
    %       w       compressor speed in rad/s
    %       p_out   ooutput pressure in Pa
    % 
    % Additional output option:
    % [W, T_out, torque, add_output] = FCmodel_Suh.compressor(...)
    %   'add_output' contains values that are used internally in
    %   this function (This option is made for debugging purposes).


    %Avoid NaNs and Inf in case w = 0
    if isnumeric(w) && w == 0
        W = 0;
        T_out = T_in;
        torque = 0;
        return;
    end
    
    % (Pukrushpan, Control of Fuel Cell Power Sytems, page 17 f)
    
    %Constants differing from rest of model
    R_a = 286.9; %J/(kg*K); air gas constant
    rho_a = 1.23; %kg/m^3; Air density
    
    %K; ideal and real Temperature increase by compressor
    T_inc_ideal = T_in * ((p_out / p_in)^((p.gamma - 1) / p.gamma) - 1);
    T_inc_real = T_inc_ideal / p.eta_cp;
    
    delta = p_in / 101325; %1; Pressure relative to 1 atm
    sTheta = sqrt(T_in / 288); %1; Temperature relative to 288 K
    w_cr = w / sTheta; %rad/s
    U_c = w_cr * p.r_c; %m/s; Blade tip speed
    Psi = p.C_p * T_inc_ideal / (U_c^2 * 0.5); %1; head parameter (compressor work)
    
    speed_sound = sqrt(p.gamma * R_a * T_in); %m/s; speed of sound in air
    M = U_c / speed_sound; %1; mach number
    
    %coefficients defined by (Pukrushpan, Control of Fuel Cell Power Sytems, page 19)
    a4 = -3.69906e-5;
    a3 = 2.70399e-4;
    a2 = -5.36235e-4;
    a1 = -4.63685e-5;
    a0 = 2.21195e-3;
    b2 = 1.76567;
    b1 = -1.34837;
    b0 = 2.44419;
    c5 = -9.78755e-3;
    c4 = 0.10581;
    c3 = -0.42937;
    c2 = 0.80121;
    c1 = -0.68344;
    c0 = 0.43331;
    
    Phi_max =            a4 * M^4 + a3 * M^3 + a2 * M^2 + a1 * M + a0; %1;
    beta    =                                  b2 * M^2 + b1 * M + b0; %1;
    Psi_max = c5 * M^5 + c4 * M^4 + c3 * M^3 + c2 * M^2 + c1 * M + c0; %1;
    
    % Normalized compressor flow rate
    Phi = Phi_max * ( 1 - exp(beta * (Psi / Psi_max - 1)) );

    %Flow
    W_cr = Phi * rho_a * pi * p.r_c^2 * U_c;
    W = W_cr * delta / sTheta;

    % Output temperature
    T_out = T_in + T_inc_real; %K;

    %torque
    P = p.C_p * T_inc_real * W; %J/s=W; Power
    torque = P / w; %Nm; Torque

end
