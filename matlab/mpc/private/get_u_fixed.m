function u = get_u_fixed(fixed_inputs, parameters, param_values, N)

    u = zeros(length(fixed_inputs), N);

    for i = 1:length(fixed_inputs)
        f = fixed_inputs(i);

        if f.value.is_numeric()
            u(i, :) = f.value.get_numeric_value();
            continue
        end

        pname = f.value.get_parameter_name();

        idx = find_signal(parameters, pname);

        value = param_values(idx);

        if length(value) == 1
            u(i, :) = value;
        else
            u(i, :) = value(1:end-1);
        end
    end
end
