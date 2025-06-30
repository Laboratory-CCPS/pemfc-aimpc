function [p_interp, W_interp] = get_W_caOut_lookup_table(p)
    % From equations 2.15 and 2.16
    c1 = (p.gamma - 1) / p.gamma;
    c2 = p.C_D * p.A_T / sqrt(p.R * p.T_st);

    %normal flow:
    W_normal = @(p_cathode) ...
        c2 * p_cathode * (p.p_atm / p_cathode)^(1 / p.gamma) ...
        * sqrt(2 / c1) * sqrt(1 - (p.p_atm / p_cathode)^c1);
    
    %choked flow (linear):
    W_choked = @(p_cathode) ...
        c2 * p_cathode ...
        * sqrt(p.gamma) ...
        * sqrt( (2 / (p.gamma + 1))^((p.gamma + 1)/(p.gamma - 1)) );

    % Critical Pressure
    % when p_cathode < p_crit, use W_normal
    p_crit = p.p_atm / ( (2 / (p.gamma + 1))^(1 / c1) ); %1.9180 bar
    
    %Create interpolation table for W_normal
    p_ca = linspace(p.p_atm, p_crit, 100);
    
    W1 = zeros(size(p_ca));
    for ii = 1:numel(p_ca)
        W1(ii) = W_normal(p_ca(ii));
    end
    
    %For W_choked, just use last value of W_normal and some other
    %value sinc W_choked is linear
    p_end = 10 * p.p_atm;

    %Combine segments to interpolation table
    p_interp = [0, p_ca, p_end          ];
    W_interp = [0, W1,   W_choked(p_end)];
end
