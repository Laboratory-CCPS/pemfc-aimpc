classdef FixedValue

    properties
        numeric_value = []
        parameter_name = ''
    end
    
    methods (Hidden)
        function obj = FixedValue(numeric_value, parameter_name)
            obj.numeric_value = numeric_value;
            obj.parameter_name = parameter_name;
        end
    end

    methods(Static)
        function obj = Parameter(name)
            obj = FixedValue([], name);
        end

        function obj = Numeric(value)
            obj = FixedValue(value, '');
        end
    end

    methods
        function result = is_numeric(obj)
            result = ~isempty(obj.numeric_value);
        end

        function value = get_numeric_value(obj)
            value = obj.numeric_value;
        end

        function name = get_parameter_name(obj)
            name = obj.parameter_name;
        end

    end
end
