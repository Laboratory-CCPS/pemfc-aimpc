function create_inequality_constraints(constraints, scalings, x_names, y_names, x, y, y_ft, opti)

    N = size(x, 2) - 1;

    for i = 1:length(constraints)
        name = constraints(i).signal;

        scaling = scalings.(name);
        
        maxval = scaling.scale(constraints(i).max);
        minval = scaling.scale(constraints(i).min);

        if constraints(i).feedthrough
            [expr, kind] = get_expr(name, x_names, y_names, x, y_ft);
        else
            [expr, kind] = get_expr(name, x_names, y_names, x, y);
        end

        if strcmp(kind, 'output') && constraints(i).feedthrough
            k0 = 1;
            kend = N;
        else
            k0 = 2;
            kend = N + 1;
        end

        for k = k0:kend
            if ~isinf(minval) && ~isinf(maxval)
                opti.subject_to(minval <= expr(k) <= maxval); %#ok<CHAIN>
            elseif ~isinf(minval)
                opti.subject_to(minval <= expr(k));
            elseif ~isinf(maxval)
                opti.subject_to(expr(k) <= maxval);
            end
        end
    end
end
