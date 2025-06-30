function [Data] = detransform_data(Data_transformed,transform_info)


if strcmp('none',transform_info.method) % no transformation
    Data = Data_transformed;

end

if strcmp('center',transform_info.method) % zero mean
    sz = size(Data_transformed);
    Data = Data_transformed + repmat(transform_info.mean_data,sz(1),1);

end

if strcmp('zscore',transform_info.method) % data have zero mean and std of 1
    sz = size(Data_transformed);
    mean_data = transform_info.mean_data;
    std_data = transform_info.std_data;

    mean_data_mat = repmat(mean_data,sz(1),1);
    std_data_mat = repmat(std_data,sz(1),1);

    Data = Data_transformed.*std_data_mat + mean_data_mat;

end

if strcmp('rescale',transform_info.method)
    sz = size(Data_transformed);
    min_data = repmat(transform_info.min_data,sz(1),1);
    max_data =  repmat(transform_info.max_data,sz(1),1);
    a = transform_info.lb;
    b = transform_info.ub;

    Data =( (Data_transformed - a)./(b-a) ).*(max_data-min_data)+min_data;

end

end