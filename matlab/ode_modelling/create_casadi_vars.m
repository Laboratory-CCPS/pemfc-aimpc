function cvars = create_casadi_vars(vars)

    for i = 1:length(vars)
        v = vars{i};

        cvars.(v) = casadi.MX.sym(v);
    end

end
