function model = scale_(model, direction_scale)

    if isempty(model.scalings)
        error("model doesn't provide scalings");
    end

    scalings = model.scalings;

    scaled_names = fieldnames(scalings);

    model.dx = casadi.MX(model.dx);
    model.y = casadi.MX(model.y);

    x_names = casadi_vars_to_str(model.states);
    u_names = casadi_vars_to_str(model.inputs);
    y_names = model.y_names;

    for i = 1:length(x_names)
        name = x_names{i};

        if ismember(name, scaled_names)
            if direction_scale
                model.dx(i) = scalings.(name).scale_derivate(model.dx(i));
            else
                model.dx(i) = scalings.(name).unscale_derivate(model.dx(i));
            end
        end
    end

    for i = 1:length(y_names)
        name = y_names{i};

        if ismember(name, scaled_names)
            if direction_scale
                model.y(i) = scalings.(name).scale(model.y(i));
            else
                model.y(i) = scalings.(name).unscale(model.y(i));
            end
        end
    end

    all_names = [x_names; u_names];
    all_vars = [model.states; model.inputs];

    no_scaling_vars = setdiff(all_names, scaled_names);

    if ~isempty(no_scaling_vars)
        warning('no scaling for the following signals given:\n    %s', strjoin(no_scaling_vars, ', '));
    end

    for i = 1:length(all_names)
        name = all_names{i};
        var = all_vars(i);

        scaling = scalings.(name);

        if direction_scale
            expr = scaling.unscale(var);
        else
            expr = scaling.scale(var);
        end

        model.dx = casadi.substitute(model.dx, var, expr);
        model.y = casadi.substitute(model.y, var, expr);
    end

    model.dx = model.dx.simplify();
    model.y = model.y.simplify();

end
