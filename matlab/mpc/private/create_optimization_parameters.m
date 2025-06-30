function [x0, params] = create_optimization_parameters(N_steps, nx, parameters, opti)

    x0 = opti.parameter(nx);

    params = struct();

    for i = 1:length(parameters)
        p = parameters(i);

        if ~p.is_vector
            pdim = 1;
        else
            pdim = N_steps + 1;
        end

        params.(p.signal) = opti.parameter(pdim);
    end

end
