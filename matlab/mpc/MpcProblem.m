classdef MpcProblem

    properties
        model

        u_names
        x_names
        y_names

        free_inputs = struct('signal', {}, 'min', {}, 'max', {}, 'init', {})
        fixed_inputs = struct('signal', {}, 'value', {})
        constraints = struct('signal', {}, 'min', {}, 'max', {}, 'feedthrough', {})
        costs_quadratic = struct('signal', {}, 'ref', {}, ...
            'weight_stage', {}, 'weight_end', {}, 'scaled_weights', {}, 'feedthrough', {})
        costs_quadratic_diffquot = struct('signal', {}, 'quot', {}, 'ts_order', {}, ...
            'weight', {}, 'scaled_weight', {}, 'feedthrough', {}, 'prev_value', {})
        costs_linear = struct('signal', {}, ...
            'weight_stage', {}, 'weight_end', {}, 'scaled_weights', {}, 'feedthrough', {})

        parameters = struct('signal', {}, 'is_vector', {}, 'scaled_like', {});
    end
    
    properties (Dependent)
        nu_free
    end

    methods
        function obj = MpcProblem(model)
            obj.model = model;
            obj.u_names = casadi_vars_to_str(obj.model.inputs);
            obj.x_names = casadi_vars_to_str(obj.model.states);
            obj.y_names = obj.model.y_names;            

        end

        function value = get.nu_free(obj)
            value = length(obj.free_inputs);
        end

        function obj = add_free_input(obj, name, min, max, init)
            if (nargin < 5), init = nan; end

            [~, id] = ismember(name, obj.u_names);

            if id == 0
                error('"%s" is not an input of the model', name);
            end

            idx = find_signal(obj.free_inputs, name);

            if idx == 0
                obj.free_inputs(end + 1) = struct('signal', name, 'min', min, 'max', max, 'init', init);
            else
                warning("'%s' was already defined as free input. " + ...
                        "Updating its parameters, but the previous index of the input remains valid.");
            
                obj.free_inputs(idx) = struct('signal', name, 'min', min, 'max', max, 'init', init);
            end
        end

        function obj = add_fixed_input(obj, name, value)
            if ~ismember(name, obj.u_names)
                error("'%s' is nont an input of the model", name);
            end

            if find_signal(obj.free_inputs, name) ~= -1
                warning("'%s' is already a free input. The fixed value will be ignored.");
            end

            idx = find_signal(obj.fixed_inputs, name);

            if idx == 0
                obj.fixed_inputs(end + 1) = struct('signal', signal, 'value', value);
            else
                obj.fixed_inputs(idx) = struct('signal', signal, 'value', value);
            end
        end

        function obj = add_fixed_inputs(obj, fixed_inputs)
            names = fieldnames(fixed_inputs);

            for i = 1:length(names)
                obj = obj.add_fixed_input(names{i}, fixed_inputs.(names{i}));
            end
        end

        function obj = add_constraint(obj, name, min, max, varargin)
            ip = inputParser();
            ip.KeepUnmatched = false;
            ip.addParameter("feedthrough", [], @islogical);
            ip.parse(varargin{:});
            feedthrough = ip.Results.feedthrough;

            if ismember(name, obj.u_names)
                error("'%s' is an input. Constraints on inputs must be specified with add_free_input", name);
            end

            if min > max
                error("invlaid constraint: max is smaller than min");
            end

            feedthrough = obj.check_signal_(name, feedthrough);
            
            for i = 1:length(obj.constraints)
                if strcmp(obj.constraints(i).signal, name) && (obj.constraints(i).feedthrough == feedthrough)
                    warning("For '%s' a constraint with the same feedthrough value was already defined and gets overwritten now.", ...
                        name);
        
                    obj.constraints(i) = struct('signal', name, 'min', min, 'max', max, 'feedthrough', feedthrough);
                    return;
                end
            end

            obj.constraints(end + 1) = struct('signal', name, 'min', min, 'max', max, 'feedthrough', feedthrough);
        end

        function obj = add_quadratic_cost(obj, name, ref, varargin)
            ip = inputParser();
            ip.KeepUnmatched = false;
            ip.addParameter("weight_stage", 0, @isnumeric);
            ip.addParameter("weight_end", 0, @isnumeric);
            ip.addParameter("feedthrough", [], @islogical);
            ip.addParameter("scaled_weights", false, @islogical);
            ip.parse(varargin{:});
            weight_stage = ip.Results.weight_stage;
            weight_end = ip.Results.weight_end;
            feedthrough = ip.Results.feedthrough;
            scaled_weights = ip.Results.scaled_weights;

            feedthrough = obj.check_signal_(name, feedthrough);

            [ref, param] = obj.check_fixed_value_(ref, name);
    
            if ~isempty(param)
                obj.parameters(end + 1) = param;
            end
            

            cost = struct('signal', name, 'ref', ref, ...
                'weight_stage', weight_stage, 'weight_end', weight_end, ...
                'scaled_weights', scaled_weights, 'feedthrough', feedthrough);

            for i = 1:length(obj.costs_quadratic)
                if strcmp(obj.costs_quadratic(i).signal, name) && (obj.costs_quadratic(i).feedthrough == feedthrough)
                    warning("For '%s' a quadratic cost with the same feedthrough value was already defined and gets overwritten now.", ...
                        name);
               
                    obj.costs_quadratic(i) = cost;
                    return;
                end
            end
            obj.costs_quadratic(end + 1) = cost;

        end

        function obj = add_quadratic_diffquot_cost(obj, name, diff_quot, varargin)
            ip = inputParser();
            ip.KeepUnmatched = false;
            ip.addParameter("ts_order", 0, @isnumeric);
            ip.addParameter("weight", 0, @isnumeric);
            ip.addParameter("feedthrough", [], @islogical);
            ip.addParameter("scaled_weight", false, @islogical);
            ip.addParameter("prev_value", []);
            ip.parse(varargin{:});
            ts_order = ip.Results.ts_order;
            weight = ip.Results.weight;
            feedthrough = ip.Results.feedthrough;
            scaled_weight = ip.Results.scaled_weight;
            prev_value = ip.Results.prev_value;

            feedthrough = obj.check_signal_(name, feedthrough);

            if abs(sum(diff_quot)) > 1e-12
                warning("different quotient factors for '%s' don't sum up to 0", name);
            end

            if ~isempty(prev_value)
                [prev_value, param] = obj.check_fixed_value_(prev_value, name);

                if ~isempty(param)
                    obj.parameters(end + 1) = param;
                end
            end

            cost = struct('signal', name, ...
                'quot', diff_quot, 'ts_order', ts_order, ...
                'weight', weight, 'scaled_weight', scaled_weight, ...
                'feedthrough', feedthrough, 'prev_value', prev_value);

            for i = 1:length(obj.costs_quadratic_diffquot)
                if strcmp(obj.costs_quadratic_diffquot(i).signal, name) ...
                        && (obj.costs_quadratic_diffquot(i).feedthrough == feedthrough)
                    warning("For '%s' a quadratic diff quot cost with the same feedthrough value was already defined and gets overwritten now.", ...
                        name);

                    obj.costs_quadratic_diffquot(i) = cost;
                    return;
                end
            end

            obj.costs_quadratic_diffquot(end + 1) = cost;

        end

        function obj = add_linear_cost(obj, name, varargin)
            ip = inputParser();
            ip.KeepUnmatched = false;
            ip.addParameter("weight_stage", 0, @isnumeric);
            ip.addParameter("weight_end", 0, @isnumeric);
            ip.addParameter("feedthrough", [], @islogical);
            ip.addParameter("scaled_weights", false, @islogical);
            ip.parse(varargin{:});
            weight_stage = ip.Results.weight_stage;
            weight_end = ip.Results.weight_end;
            feedthrough = ip.Results.feedthrough;
            scaled_weights = ip.Results.scaled_weights;

            feedthrough = obj.check_signal_(name, feedthrough);

            cost = struct('signal', name, ...
                'weight_stage', weight_stage, 'weight_end', weight_end, ...
                'scaled_weights', scaled_weights, 'feedthrough', feedthrough);

            for i = 1:length(obj.costs_linear)
                if strcmp(obj.costs_linear(i).signal, name) && (obj.costs_linear(i).feedthrough == feedthrough)
                    warning("For '%s' a quadratic cost with the same feedthrough value was already defined and gets overwritten now.", ...
                        name);

                    obj.costs_linear(i) = cost;
                    return;
                end
            end

            obj.costs_linear(end + 1) = cost;

        end

        function cons = get_all_constraints(obj)
            cons = struct();

            for i = 1:length(obj.constraints)
                c = obj.constraints(i);
                cons.(c.signal).min = c.min;
                cons.(c.signal).max = c.max;
            end

            for i = 1:length(obj.free_inputs)
                u = obj.free_inputs(i);
                cons.(u.signal).min = u.min;
                cons.(u.signal).max = u.max;
            end
        end
    end

    methods(Hidden)
        function [value, param] = check_fixed_value_(obj, value, ref_signal)
            if isnumeric(value)
                value = FixedValue.Numeric(value);
                param = [];
                return;
            end

            name = strtrim(value);

            if isempty(value) || (value(1) ~= '#')
                error("Fixed value must be numerical or start with '#'");
            end

            name = name(2:end);

            is_vector = (length(value) > 2) && strcmp(value(end-1:end), '[]');

            if is_vector
                name = name(1:end-2);
            end

            if ismember(name, obj.x_names)
                error("parameter name '%s' is already a state", name);
            elseif ismember(name, obj.u_names)
                error("parameter name '%s' is already an input", name);
            elseif ismember(name, obj.y_names)
                error("parameter name '%s' is already an output", name);
            elseif ismember(name, {'x0', 'u', 'lam_g'})
                error("parameter name '%s' is a reserved name", pname);
            end

            idx = find_signal(obj.parameters, name);

            if (idx == 0)
                param = struct('signal', name, 'is_vector', is_vector, 'scaled_like', ref_signal);
            else
                error("'%s' is already used as a parameter name");
            end
            
            value = FixedValue.Parameter(name);
        end

        function feedthrough = check_signal_(obj, name, feedthrough)
            if ismember(name,  obj.u_names)
                if ~isempty(feedthrough)
                    error("The parameter 'feedthrough' can only used with outputs, but" ...
                          + " '%s' is an input.", name);
                end
                feedthrough = true;

            elseif ismember(name, obj.x_names)
                if ~isempty(feedthrough)
                    error("The parameter 'feedthrough' can only used with or outputs, but" ...
                          + " '%s' is a state.", name);
                end
                feedthrough = false;

            elseif ismember(name, obj.model.y_names)
                if isempty(feedthrough)
                    feedthrough = false;
                end

            else
                error("'%s' is not a signal of the model.", name);
            end
        end
    end
end
