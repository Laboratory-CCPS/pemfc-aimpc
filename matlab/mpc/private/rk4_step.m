function x = rk4_step(f, x0, u, Ts, substeps)

    dt = Ts / substeps;
    
    x = x0;

    for j = 1:substeps
        k1 = f(x, u);
        k2 = f(x + dt / 2 * k1, u);
        k3 = f(x + dt / 2 * k2, u);
        k4 = f(x + dt * k3, u);
        x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    end

end
