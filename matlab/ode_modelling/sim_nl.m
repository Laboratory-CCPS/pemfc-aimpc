function res = sim_nl(model, Ts, x0, u)

    simulator = Simulator(model, Ts);
    
    N_sim = size(u, 2);
    
    % initialize simulation parameters
    x = NaN(length(x0), N_sim + 1);
    x(:, 1) = x0;
    
    y0 = full(simulator.f_out(x0, u(:, 1).'));
    y = NaN(length(y0), N_sim + 1);
    y_ft = NaN(length(y0), N_sim + 1);
    y(:, 1) = y0;
        
    t = (0:N_sim) * Ts;

    for i = 1:N_sim
        % current states/inputs
        x_i = x(:, i);
        u_i = u(:, i);
    
        % integrate for one time step
        try
            [x_next, y_next, y_ft_i] = simulator.sim_step(x_i, u_i);
        catch
            warning("Simulation failed at step %i (t = %g s)", i, i * Ts);
            break;
        end

        % update
        x(:, i + 1) = x_next;    
        y(:, i + 1) = y_next;
        y_ft(:, i) = y_ft_i;
    end

    y_ft(:, end) = y(:, end);

    res.t = t;
    res.u = [u, nan(size(u, 1), 1)];
    res.x = x;
    res.y = y;
    res.y_ft = y_ft;
    res.u_names = casadi_vars_to_str(model.inputs);
    res.x_names = casadi_vars_to_str(model.states);
    res.y_names = model.y_names;

    res.scalings = model.scalings;
    res.scaled = model.scaled;

    res.meta.discrete = simulator.meta.discrete;
    res.meta.Ts = Ts;
    res.desc = '';

end
