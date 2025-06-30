function p_sat = satPressure(T)
    %Vapor saturation pressure from temperature T in Kelvin
    switch T
        case celsius2kelvin(25)
            p_sat = 0.0317395e5;
        case celsius2kelvin(80)
            p_sat = 0.47373e5;
        case celsius2kelvin(70)
            p_sat = 0.31176e5;
        otherwise
            error('Function satPressure cant evaluate at Temperature %f K', T);
    end
            
end