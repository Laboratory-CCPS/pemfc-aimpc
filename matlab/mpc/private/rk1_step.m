function x = rk1_step(f, x0, u, Ts, substeps)

    dt = Ts / substeps;
    
    x = x0;

    for j = 1:substeps
        x = x + dt * f(x, u);
    end

end
