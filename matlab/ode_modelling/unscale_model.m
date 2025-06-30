function model = unscale_model(model)

    if ~model.scaled
        warning("model already unscaled");
        return;
    end

    model = scale_(model, false);

    model.scaled = false;
end
