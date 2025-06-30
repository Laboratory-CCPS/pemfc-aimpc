function x = idas_step(F, x0, u, substeps)

    x = x0;

    for j = 1:substeps
        r = F('x0', x, 'p', u);
        x = r.xf;
    end

end
