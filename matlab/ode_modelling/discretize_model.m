function dmodel = discretize_model(model, Ts, method, varargin)

    ip = inputParser();
    ip.KeepUnmatched = false;
    ip.addParameter("Substeps", 1, @isPositiveIntegerValuedNumeric);
    ip.parse(varargin{:});

    substeps = ip.Results.Substeps;

    assert(ismember(method, {'rk1', 'rk4'}));

    f = casadi.Function('f', {model.states, model.inputs}, {model.dx});

    x0 = arrayfun(@(v) casadi.MX.sym(str(v)), model.states, 'UniformOutput', false);
    x0 = vertcat(x0{:});

    u = arrayfun(@(v) casadi.MX.sym(str(v)), model.inputs, 'UniformOutput', false);
    u = vertcat(u{:});
    
    dt = Ts / substeps;    
    x = x0;

    switch method
        case 'rk1'
            for j = 1:substeps
                x = x + dt * f(x, u);
            end

        case 'rk4'           
            for j = 1:substeps
                k1 = f(x, u);
                k2 = f(x + dt / 2 * k1, u);
                k3 = f(x + dt / 2 * k2, u);
                k4 = f(x + dt * k3, u);
                x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6;
            end

        otherwise
            assert(false);
    end

    dmodel.states = x0;
    dmodel.inputs = u;
    dmodel.Ts = Ts;
    dmodel.x_next = x;
    dmodel.y = casadi.substitute(model.y, [model.states; model.inputs], [dmodel.states; dmodel.inputs]);
    dmodel.y_names = model.y_names;

    dmodel.scalings = model.scalings;
    dmodel.scaled = model.scaled;

end
