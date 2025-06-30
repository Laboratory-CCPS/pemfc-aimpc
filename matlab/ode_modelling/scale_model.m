function model = scale_model(model)

    if model.scaled
        warning("model already scaled");
        return;
    end

    model = scale_(model, true);

    model.scaled = true;
end
