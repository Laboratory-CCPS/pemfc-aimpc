function Data_noNaN = remove_row_with_nancolumn(Data,which_column)

rowsWithNaN = cellfun(@(x) any(isnan(x(:))), Data(:,which_column)); % we only look into which_column column

Data(rowsWithNaN,:) = [];
Data_noNaN = Data;


end