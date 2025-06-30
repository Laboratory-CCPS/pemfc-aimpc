function res = build_sim_result_from_data(model, t, u, x, y, y_ft)

    assert(isvector(t));

    nt = length(t);
    nx = length(model.states);
    nu = length(model.inputs);
    ny = length(model.y_names);

    assert(size(x, 1) == nx);
    assert(size(x, 2) == nt);

    assert(size(y, 1) == ny);
    assert(size(y, 2) == nt);

    assert(isempty(y_ft) || size(y_ft, 1) == ny);
    assert(isempty(y_ft) || size(y_ft, 2) == nt);

    assert(size(u, 1) == nu);
    assert(size(u, 2) == nt);
    
    res.t = t;
    res.u = u;
    res.x = x;
    res.y = y;
    res.y_ft = y_ft;
    res.u_names = casadi_vars_to_str(model.inputs);
    res.x_names = casadi_vars_to_str(model.states);
    res.y_names = model.y_names;

    res.scalings = model.scalings;
    res.scaled = model.scaled;

    res.desc = '';

end
