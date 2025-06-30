%%
function [Data_transformed,transform_info] = transform_data(Data,method,varargin)
% see also normalize() function from matlab
% sadly this function does not have an inbuilt way to detransform the data
% again... so we do it here in an extra function

% the function can be used together with the function "detransform_data"

% if only "Data" and "method" are provided the function transforms the
% input data according to the method described in "method" (for example
% based on the mean of the provided data)

% if you provide a certain transformation rule transform_info =
% varargin{1}; 
%then data are transformed according to the provided rules (for example
%based on some provided mean value)

% its easier to understand if you directly look at the code

% the function outputs the transformed data and a transform_info struct,
% which stores all information to detransform the data or to transform new
% data point always in the same way. (if used as varargin{1} = transform_info)


if nargin == 2 % transform based on provided data
    if strcmp('none',method) % no transformation
        Data_transformed = Data;
        transform_info.method = 'none';

    end

    if strcmp('center',method) % data have zero mean
        sz = size(Data);
        mean_data = mean(Data);
        Data_transformed = Data - repmat(mean_data,sz(1),1);
        transform_info.method = 'center';
        transform_info.mean_data = mean_data;

    end

    if strcmp('zscore',method) % data have zero mean and std of 1
        sz = size(Data);
        mean_data = mean(Data);
        std_data = std(Data);
        mean_data_mat = repmat(mean_data,sz(1),1);
        std_data_mat = repmat(std_data,sz(1),1);
        Data_transformed = (Data - mean_data_mat)./std_data_mat;
        transform_info.method = 'zscore';
        transform_info.mean_data = mean_data;
        transform_info.std_data = std_data;

    end

    if strcmp('rescale',method)% for now scale data between -1 and 1 
        sz = size(Data);
        min_d = min(Data);
        max_d = max(Data);
        min_data = repmat(min_d,sz(1),1);
        max_data = repmat(max_d,sz(1),1);
        a = -1;
        b = 1;
        Data_transformed = a + ((Data-min_data)./(max_data-min_data)).*(b-a);
        transform_info.method = 'rescale';
        transform_info.min_data = min_d;
        transform_info.max_data = max_d;
        transform_info.lb = a;
        transform_info.ub = b;

    end

elseif nargin == 3 % transform according to provided rule
    transform_info = varargin{1};
    if strcmp('none',transform_info.method) % no transformation
        Data_transformed = Data;
        %transform_info.method = 'none';

    end

    if strcmp('center',transform_info.method) % data have zero mean
        sz = size(Data);
        Data_transformed = Data - repmat(transform_info.mean_data,sz(1),1);

    end

    if strcmp('zscore',transform_info.method) % data have zero mean and std of 1
        sz = size(Data);
        mean_data_mat = repmat(transform_info.mean_data,sz(1),1);
        std_data_mat = repmat(transform_info.std_data,sz(1),1);
        Data_transformed = (Data - mean_data_mat)./std_data_mat;

    end

    if strcmp('rescale',transform_info.method)% for now scale data between -1 and 1 
        sz = size(Data);
        min_data = repmat(transform_info.min_data,sz(1),1);
        max_data = repmat(transform_info.max_data,sz(1),1);
        a = transform_info.lb;%-1;
        b = transform_info.ub;% 1;
        Data_transformed = a + ((Data-min_data)./(max_data-min_data)).*(b-a);

    end



end %if nargin


end %fct