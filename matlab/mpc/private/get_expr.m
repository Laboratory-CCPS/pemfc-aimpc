function [expr, kind] = get_expr(name, x_names, y_names, x, y)

    idx = find_element(x_names, name);

    if idx ~= 0
        expr = x(idx, :);
        kind = 'state';
        return;
    end

    idx = find_element(y_names, name);

    if idx ~= 0
        expr = y(idx, :);
        kind = 'output';
        return;
    end
    
    assert(false);
    
end
