function [u_free, xk] = create_optimization_variables(N, nu, nx, multiple_shooting, opti)

    % the rows of u correspond to the free variables in the order
    % in which they were added by add_free_input, i.e. the order
    % is given by the order of the fields of obj.u_free_inputs
    u_free = opti.variable(nu, N);

    if multiple_shooting
        xk = opti.variable(nx, N);
    else
        xk = [];
    end

end
