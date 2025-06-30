function res = unscale_sim_results(res)

    if ~res.scaled
        warning("data allready unscaled");
        return;
    end

    res.u = unscale_data(res.scalings, res.u_names, res.u);
    res.x = unscale_data(res.scalings, res.x_names, res.x);
    res.y = unscale_data(res.scalings, res.y_names, res.y);

    if ismember('y_ft', fieldnames(res))
        res.y_ft = unscale_data(res.scalings, res.y_names, res.y_ft);
    end

    res.scaled = false;
    
end
