function create_input_constraints(free_inputs, scalings, u_free, opti)

    N = size(u_free, 2);

    for i = 1:length(free_inputs)
        name = free_inputs(i).signal;
        scaling = scalings.(name);

        minval = scaling.scale(free_inputs(i).min);
        maxval = scaling.scale(free_inputs(i).max);
    
        for k = 1:N
            if ~isinf(minval) && ~isinf(maxval)
                opti.subject_to(minval <= u_free(i, k) <= maxval); %#ok<CHAIN>
            elseif ~isinf(minval)
                opti.subject_to(minval <= u_free(i, k));
            elseif ~isinf(maxval)
                opti.subject_to(u_mpc(i, k) <= maxval);
            end
        end
    end   
end
