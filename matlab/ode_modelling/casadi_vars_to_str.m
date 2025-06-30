function names = casadi_vars_to_str(vars)

    names = arrayfun(@(v) str(v), vars, 'UniformOutput', false);

end
