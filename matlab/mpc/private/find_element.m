function i = find_element(cells, signal)

    for i = 1:length(cells)
        if strcmp(cells{i}, signal)
            return
        end
    end

    i = 0;
end
