function indices = get_input_indices(u_names, free_inputs, fixed_inputs)

    indices = struct('kind', {}, 'index', {});
    missing_inputs = {};

    for i = 1:length(u_names)
        name = u_names{i};

        idx = find_signal(free_inputs, name);

        if idx ~= 0
            indices(end + 1) = struct('kind', 'free', 'index', idx); %#ok<AGROW>
            continue;
        end

        idx = find_signal(fixed_inputs, name);

        if idx ~= 0
            indices(end + 1) = struct('kind', 'fixed', 'index', idx); %#ok<AGROW>
            continue;
        end

        missing_inputs{end + 1} = name; %#ok<AGROW>
    end

    if ~isempty(missing_inputs)
        error("the following input(s) are neither specified as free or as fixed: %s", ...
            strjoin(missing_inputs, ', '));
    end
end
