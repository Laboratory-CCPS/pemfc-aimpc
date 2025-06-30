function value = get_fixed_value(fv, idx, parameters, params)

    if fv.is_numeric()
        value = fv.get_numeric_value();
        return;
    end

    pname = fv.get_parameter_name();

    pidx = find_signal(parameters, pname);

    if parameters(pidx).is_vector
        value = params.(pname)(idx);
    else
        value = params.(pname);
    end

end
