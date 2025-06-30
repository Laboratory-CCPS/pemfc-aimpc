classdef Scaling
    properties(SetAccess=private)
        factor
        offset
    end

    methods
        function obj = Scaling(factor, offset)
            obj.factor = factor;

            if (nargin == 1)
                obj.offset = 0;
            else
                obj.offset = offset;
            end
        end

        function unscale = get_unscale_scaling(obj)
            unscale = Scaling(1 / obj.factor, -obj.offset / obj.factor);
        end

        function value = scale(obj, value)
            value = (value - obj.offset) / obj.factor;
        end

        function value = unscale(obj, value)
            value = value * obj.factor + obj.offset;
        end

        function value = scale_derivate(obj, value)
            value = value / obj.factor;
        end

        function value = unscale_derivate(obj, value)
            value = value * obj.factor;
        end
    end

    methods(Static)
        function scaling = FromRange(min, max)
            factor = (max - min) / 2;
            offset = (max + min) / 2;

            scaling = Scaling(factor, offset);
        end
    end
end
