function u = build_input_matrix(u_names, free_inputs, fixed_inputs, parameters, u_free, params)

    N = size(u_free, 2);

    indices = get_input_indices(u_names, free_inputs, fixed_inputs);

    u = cell(1, N);

    for k = 1:N
        uk = cell(1, length(u_names));

        for i = 1:length(indices)
            ind = indices(i);

            if strcmp(ind.kind, 'free')
                uk{i} = u_free(ind.index, k);
            else
                uk{i} = get_fixed_value(fixed_inputs(ind.index), k, parameters, params);
            end
        end

        u{k} = vertcat(uk{:});
    end

    u = horzcat(u{:});

end
