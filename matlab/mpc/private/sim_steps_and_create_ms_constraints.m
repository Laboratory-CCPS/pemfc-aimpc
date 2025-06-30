function [x, y, y_ft] = sim_steps_and_create_ms_constraints(...
    model, x0, u, Ts, discretization, substeps, xk, opti)

    multiple_shooting = ~isempty(xk);
    N = size(u, 2);

    if strcmp(discretization, 'idas')
        dae = struct('x', model.states, 'p', model.inputs, 'ode', model.dx);
        F = casadi.integrator('F', 'idas', dae, 0, Ts / substeps);                
    else
        f_dx = casadi.Function('f', {model.states, model.inputs}, {model.dx});
    end

    f_out = casadi.Function('outputs', {model.states, model.inputs}, {model.y});
    
    x = cell(N + 1, 1);
    x{1} = x0;

    y = cell(N + 1, 1);
    y{1} = f_out(x0, u(:, 1));

    y_ft = cell(N + 1, 1);
    y_ft{1} = f_out(x0, u(:, 1));

    for i = 1:N
        switch discretization
            case 'rk1'
                x_i = rk1_step(f_dx, x{i}, u(:, i), Ts, substeps);
            case 'rk4'
                x_i = rk4_step(f_dx, x{i}, u(:, i), Ts, substeps);
            case 'idas'
                x_i = idas_step(F, x{i}, u(:, i), substeps);
            otherwise
                assert(false);
        end

        if multiple_shooting
            opti.subject_to(xk(:, i) == x_i);
            x{i + 1} = xk(:, i);
        else
            x{i + 1} = x_i;
        end

        y{i + 1} = f_out(x{i + 1}, u(:, i));

        if i < N
            y_ft{i + 1} = f_out(x{i + 1}, u(:, i + 1));
        else
            y_ft{i + 1} = y{i + 1};
        end
    end

    x = horzcat(x{:});
    y = horzcat(y{:});
    y_ft = horzcat(y_ft{:});

end
