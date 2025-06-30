function data = scale_signals(scalings, data)

    scaled_names = fieldnames(scalings);

    for i = 1:length(scaled_names)
        name = scaled_names{i};

        if isfield(data, name)
            data.(name) = scalings.(name).scale(data.(name));
        end        
    end
end
