function i = find_signal(list, signal)

    for i = 1:length(list)
        if strcmp(list(i).signal, signal)
            return
        end
    end

    i = 0;
end
