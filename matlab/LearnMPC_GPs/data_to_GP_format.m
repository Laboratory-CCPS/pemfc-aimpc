function Data_GP = data_to_GP_format(Data,which_u)

dim_x_state = length(Data{1,1});


InitalState = reshape(cell2mat(Data(:,1)), dim_x_state, []).';
reference = cell2mat(Data(:,2));
Data_GP.X = [InitalState, reference];
Data_GP.Y = cellfun(@(x) x(which_u, 1), Data(:, 4));


end