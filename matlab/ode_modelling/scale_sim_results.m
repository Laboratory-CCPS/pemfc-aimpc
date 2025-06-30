function res = scale_sim_results(res)

    if res.scaled
        warning("data allready unscaled");
        return;
    end

    res.u = scale_data(res.scalings, res.u_names, res.u);
    res.x = scale_data(res.scalings, res.x_names, res.x);
    res.y = scale_data(res.scalings, res.y_names, res.y);

    if ismember('y_ft', fieldnames(res))
        res.y_ft = scale_data(res.scalings, res.y_names, res.y_ft);
    end

    res.sclaed = true;
    
end
