function model = get_model(model_fct, p)

    [vars, scalings] = model_fct();

    x = create_casadi_vars(vars.states);
    u = create_casadi_vars(vars.inputs);
    
    states = struct2cell(x);
    model.states = vertcat(states{:});

    inputs = struct2cell(u);
    model.inputs = vertcat(inputs{:});

    [dx, y] = model_fct(x, u, p);
    
    dx = struct2cell(dx);
    dx = vertcat(dx{:});

    y_names = fieldnames(y);
    y = struct2cell(y);
    y = vertcat(y{:});

    model.dx = dx;
    model.y = y;

    model.y_names = y_names;

    model.scalings = scalings;
    model.scaled = false;

end
