function data = unscale_model_signals(model, kind, data)

    switch kind
        case {'states', 'x'}
            data = unscale_data(model.scalings, casadi_vars_to_str(model.states), data);
        case {'inputs', 'u'}
            data = unscale_data(model.scalings, casadi_vars_to_str(model.inputs), data);
        case {'outputs', 'y'}
            data = unscale_data(model.scalings, model.y_names, data);
        otherwise
            error("parameter 'kind' must be 'states', 'inputs' or 'outputs'")
    end

end
