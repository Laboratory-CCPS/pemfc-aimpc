function data = unscale_data(scalings, names, data)

    assert(length(names) == size(data, 1));

    for i = 1:length(names)
        name = names{i};

        if ~ismember(name, fieldnames(scalings))
            warning("no scaling for variable '%s' given", name);
        end

        scale = scalings.(name);

        data(i, :) = scale.unscale(data(i, :));
    end

end
