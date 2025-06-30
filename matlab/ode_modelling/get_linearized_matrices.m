function [A, B, C, D] = get_linearized_matrices(model, x, u)

    return_fcts = (nargin == 1);

    df_dx = jacobian(model.dx, model.states);    
    df_dx = casadi.Function('df_dx', {model.states, model.inputs}, {df_dx}, {'x', 'u'}, {'df_dx'});

    df_du = jacobian(model.dx, model.inputs);    
    df_du = casadi.Function('df_dx', {model.states, model.inputs}, {df_du}, {'x', 'u'}, {'df_dx'});

    dh_dx = jacobian(model.y, model.states);    
    dh_dx = casadi.Function('dh_dx', {model.states, model.inputs}, {dh_dx}, {'x', 'u'}, {'df_dx'});

    dh_du = jacobian(model.y, model.inputs);    
    dh_du = casadi.Function('dh_du', {model.states, model.inputs}, {dh_du}, {'x', 'u'}, {'df_dx'});

    if return_fcts
        A = df_dx;
        B = df_du;
        C = dh_dx;
        D = dh_du;
    else
        A = full(df_dx(x, u));
        B = full(df_du(x, u));
        C = full(dh_dx(x, u));
        D = full(dh_du(x, u));
    end
end
