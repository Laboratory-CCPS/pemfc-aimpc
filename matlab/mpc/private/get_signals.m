function s = get_signals(list)

    s = arrayfun(@(x) x.signal, list, 'UniformOutput', false);

end
