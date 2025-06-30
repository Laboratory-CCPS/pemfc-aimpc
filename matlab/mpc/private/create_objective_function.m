function create_objective_function(...
    costs_quadratic, costs_quadratic_diffquot, costs_linear, ...
    scalings, scaled, Ts, free_inputs, parameters, x_names, y_names, ...
    x, y, y_ft, u_free, params, opti)

    J = 0;

    for i = 1:length(costs_quadratic)
        J = J + get_quadratic_cost_term_(...
            costs_quadratic(i), ...
            scalings, ...
            scaled, ...
            free_inputs, ...
            parameters, ...
            x_names, ...
            y_names, ...
            x, ...
            y, ...
            y_ft, ...
            u_free, ...
            params);
    end

    for i = 1:length(costs_quadratic_diffquot)
        J = J + get_quadratic_dq_cost_term_(...
            costs_quadratic_diffquot(i), ...
            scalings, ...
            scaled, ...
            Ts, ...
            free_inputs, ...
            parameters, ...
            x_names, ...
            y_names, ...
            x, ...
            y, ...
            y_ft, ...
            u_free, ...
            params);
    end

    for i = 1:length(costs_linear)
        J = J + get_linear_cost_term_(...
            costs_linear(i), ...
            scalings, ...
            scaled, ...
            free_inputs, ...
            x_names, ...
            y_names, ...
            x, ...
            y, ...
            y_ft, ...
            u_free);
    end
    
    opti.minimize(J)

end


function J = get_quadratic_cost_term_(cost, scalings, scaled, ...
    free_inputs, parameters, x_names, y_names, x, y, y_ft, u_free, params)

    name = cost.signal;
    scaling = scalings.(name);

    N = size(u_free, 2);

    J = 0;

    idx = find_signal(free_inputs, name);

    if idx ~= 0
        if cost.weight_stage > 0
            weight = cost.weight_stage;

            if ~cost.scaled_weights && scaled
                weight = weight * scaling.factor^2;
            elseif cost.scaled_weights && ~scaled
                warning("Weight is marked as scaled, but mpc task is not scaled. No (un)scaling is applied.");
            end

            for k = 1:N
                ref_k = get_fixed_value(cost.ref, k, parameters, params);
                ref_k = scaling.scale(ref_k);

                J = J + weight * (u_free(idx, k) - ref_k)^2;
            end
        end

        if cost.weight_end > 0
            warning("ignoring end cost for input signal '%s'", name);
        end
    else
        if cost.feedthrough
            [expr, kind] = get_expr(name, x_names, y_names, x, y_ft);
        else
            [expr, kind] = get_expr(name, x_names, y_names, x, y);
        end

        if strcmp(kind, 'output') && cost.feedthrough
            k0 = 1;
            kend = N;
        else
            k0 = 2;
            kend = N;
        end

        if cost.weight_stage > 0
            weight = cost.weight_stage;

            if ~cost.scaled_weights && scaled
                weight = weight * scaling.factor^2;
            elseif cost.scaled_weights && ~scaled
                warning("Weight is marked as scaled, but mpc task is not scaled. No (un)scaling is applied.");
            end

            for k = k0:kend
                ref_k = get_fixed_value(cost.ref, k, parameters, params);
                ref_k = scaling.scale(ref_k);

                J = J + weight * (expr(k) - ref_k)^2;
            end
        end
            
        if cost.weight_end > 0
            weight = cost.weight_end;

            if ~cost.scaled_weights && scaled
                weight = weight * scaling.factor^2;
            elseif cost.scaled_weights && ~scaled
                warning("Weight is marked as scaled, but mpc task is not scaled. No (un)scaling is applied.");
            end

            ref_N = get_fixed_value(cost.ref, N + 1, parameters, params);
            ref_N = scaling.scale(ref_N);

            J = J + weight * (expr(N + 1) - ref_N)^2;
        end
    end
end


function J = get_quadratic_dq_cost_term_(cost, scalings, scaled, Ts, ...
    free_inputs, parameters, x_names, y_names, x, y, y_ft, u_free, params)

    name = cost.signal;
    scaling = scalings.(name);

    N = size(u_free, 2);
    J = 0;

    weight = cost.weight;

    if weight <= 0
        return;
    end

    if ~cost.scaled_weight && scaled
        weight = weight * scaling.factor^2;
    elseif cost.scaled_weight && ~scaled
        warning("Weight is marked as scaled, but mpc task is not scaled. No (un)scaling is applied.");
    end

    if cost.ts_order ~= 0
        weight = weight / Ts^costr.ts_order;
    end

    if isempty(cost.prev_value)
        k0 = 1;
    else
        k0 = 0;

        prev_value = get_fixed_value(cost.prev_value, 1, parameters, params);
        prev_value = scaling.scale(prev_value);
    end

    idx = find_signal(free_inputs, name);

    if idx ~= 0
        kend = N - length(cost.quot) + 1;

        for k = k0:kend
            dq = 0;

            for i = 1:length(cost.quot)
                if k == 0
                    dq = dq + cost.quot(i) * prev_value;
                else
                    dq = dq + cost.quot(i) * u_free(idx, k + i - 1);
                end
            end

            J = J + weight * dq^2;
        end
    else
        if cost.feedthrough
            [expr, kind] = get_expr(name, x_names, y_names, x, y_ft);
        else
            [expr, kind] = get_expr(name, x_names, y_names, x, y);
        end

        if strcmp(kind, 'output') && cost.feedthrough
            kend = N - length(cost.quot) + 1;
        else
            kend = N + 1 - length(cost.quot) + 1;
        end

        for k = k0:kend
            dq = 0;

            for i = 1:length(cost.quot)
                if k == 0
                    dq = dq + cost.quot(i) * prev_value;
                else
                    dq = dq + cost.quot(i) * expr(k + i - 1);
                end
            end

            J = J + weight * dq^2;
        end
    end
end


function J = get_linear_cost_term_(cost, scalings, scaled, ...
    free_inputs, x_names, y_names, x, y, y_ft, u_free)

    name = cost.signal;
    scaling = scalings.(name);

    N = size(u_free, 2);

    J = 0;

    idx = find_signal(free_inputs, name);

    if idx ~= 0
        if cost.weight_stage ~= 0
            weight = cost.weight_stage;

            if ~cost.scaled_weights && scaled
                weight = weight * scaling.factor;
            elseif cost.scaled_weights && ~scaled
                warning("Weight is marked as scaled, but mpc task is not scaled. No (un)scaling is applied.");
            end

            for k = 1:N
                J = J + weight * u_free(idx, k);
            end
        end

        if cost.weight_end ~= 0
            warning("ignoring end cost for input signal '%s'", name);
        end
    else
        if cost.feedthrough
            [expr, kind] = get_expr(name, x_names, y_names, x, y_ft);
        else
            [expr, kind] = get_expr(name, x_names, y_names, x, y);
        end

        if strcmp(kind, 'output') && cost.feedthrough
            k0 = 1;
            kend = N;
        else
            k0 = 2;
            kend = N;
        end

        if cost.weight_stage ~= 0
            weight = cost.weight_stage;

            if ~cost.scaled_weights && scaled
                weight = weight * scaling.factor;
            elseif cost.scaled_weights && ~scaled
                warning("Weight is marked as scaled, but mpc task is not scaled. No (un)scaling is applied.");
            end

            for k = k0:kend
                J = J + weight * expr(k);
            end
        end
            
        if cost.weight_end ~= 0
            weight = cost.weight_end;

            if ~cost.scaled_weights && scaled
                weight = weight * scaling.factor;
            elseif cost.scaled_weights && ~scaled
                warning("Weight is marked as scaled, but mpc task is not scaled. No (un)scaling is applied.");
            end

            J = J + weight * expr(N + 1);
        end
    end
end