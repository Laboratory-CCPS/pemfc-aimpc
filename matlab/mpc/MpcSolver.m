classdef MpcSolver

    properties
        model
        scaled
        orig_scaled

        u_names
        x_names
        y_names

        scalings

        Ts
        N

        use_lambda
        multiple_shooting
        use_opti_function

        opti
        f = []

        casadi_vars

        free_inputs
        fixed_inputs
        parameters
        mpcproblem
    end

    methods
        function obj = MpcSolver(mpcproblem, Ts, N, discretization, varargin)
            ip = inputParser();
            ip.KeepUnmatched = false;
        
            ip.addParameter("substeps", 1, @(x) isscalar(x) && (mod(x, 1) == 0) && (x > 0));
            ip.addParameter("scale_model", false, @islogical);
            ip.addParameter("use_lambda", false, @islogical);
            ip.addParameter("multiple_shooting", false, @islogical);
            ip.addParameter("use_opti_function", false, @islogical);
            ip.addParameter("verbose", true, @islogical);
            ip.addParameter("plugin_options", struct(), @isstruct);
            ip.addParameter("solver_options", struct(), @isstruct);

            ip.parse(varargin{:});

            assert(ismember(discretization, {'rk1', 'rk4', 'idas'}));
            
            substeps = ip.Results.substeps;
            scale_model_ = ip.Results.scale_model;
            use_lambda_ = ip.Results.use_lambda;
            multiple_shooting_ = ip.Results.multiple_shooting;
            use_opti_function_ = ip.Results.use_opti_function;
            verbose = ip.Results.verbose;
            plugin_opts = ip.Results.plugin_options;
            solver_opts = ip.Results.solver_options;
            
            obj.scaled = scale_model_;
            obj.orig_scaled = mpcproblem.model.scaled;

            if obj.scaled
                obj.model = scale_model(mpcproblem.model);
                obj.scalings = obj.model.scalings;
            else
                obj.model = mpcproblem.model;
                obj.scalings = struct();
            end

            obj.u_names = mpcproblem.u_names;
            obj.x_names = mpcproblem.x_names;
            obj.y_names = mpcproblem.y_names;

            all_names = [obj.u_names; obj.x_names; obj.y_names];
            missing_scalings = setdiff(all_names, fieldnames(obj.scalings));

            for i = 1:length(missing_scalings)
                obj.scalings.(missing_scalings{i}) = Scaling(1, 0);
            end

            for i = 1:length(mpcproblem.parameters)
                pname = mpcproblem.parameters(i).signal;
                scaled_like = mpcproblem.parameters(i).scaled_like;

                obj.scalings.(pname) = obj.scalings.(scaled_like);
            end

            obj.Ts = Ts;
            obj.N = N;

            obj.multiple_shooting = multiple_shooting_;
            obj.use_opti_function = use_opti_function_;
            obj.use_lambda = use_lambda_;

            obj.opti = casadi.Opti();
            
            nx = length(obj.x_names);
            nu = length(obj.u_names);

            [u_free, xk] = create_optimization_variables(N, nu, nx, multiple_shooting_, obj.opti);

            [x0, params] = create_optimization_parameters(N, nx, mpcproblem.parameters, obj.opti);

            u = build_input_matrix(...
                mpcproblem.u_names, ...
                mpcproblem.free_inputs, ...
                mpcproblem.fixed_inputs, ...
                mpcproblem.parameters, ...
                u_free, ...
                params);

            [x, y, y_ft] = sim_steps_and_create_ms_constraints(...
                obj.model, x0, u, Ts, discretization, substeps, xk, obj.opti);

            create_input_constraints(mpcproblem.free_inputs, obj.scalings, u_free, obj.opti);

            create_inequality_constraints(...
                mpcproblem.constraints, ...
                obj.scalings, ...
                mpcproblem.x_names, ...
                mpcproblem.y_names, ...
                x, ...
                y, ...
                y_ft, ...
                obj.opti);

            create_objective_function(...
                mpcproblem.costs_quadratic, ...
                mpcproblem.costs_quadratic_diffquot, ...
                mpcproblem.costs_linear, ...
                obj.scalings, ...
                obj.scaled, ...
                obj.Ts, ...
                mpcproblem.free_inputs, ...
                mpcproblem.parameters, ...
                mpcproblem.x_names, ...
                mpcproblem.y_names, ...
                x, ...
                y, ...
                y_ft, ...
                u_free, ...
                params, ...
                obj.opti);
            
            if verbose
                print_level = 5;
                print_time = 1;
            else
                print_level = 0;
                print_time = 0;
            end
            
            if ~isfield(plugin_opts, 'detect_simple_bounds')
                plugin_opts.detect_simple_bounds = true;
            end

            if ~isfield(plugin_opts, 'print_time')
                plugin_opts.print_time = print_time;
            end

            if ~isfield(solver_opts, 'print_level')
                solver_opts.print_level = print_level;
            end

            obj.opti.solver('ipopt', plugin_opts, solver_opts);
            
            if obj.use_opti_function
                x_return = x(:, 2:end);
                ps = arrayfun(@(p) params.(p.signal), mpcproblem.parameters, 'UniformOutput', false);
    
                if obj.multiple_shooting
                    if obj.use_lambda
                        obj.f = obj.opti.to_function('f', ...
                            [{x0, u_free, xk}, ps, {obj.opti.lam_g}], ...
                            {u_free, x_return, obj.opti.lam_g});
                    else
                        obj.f = obj.opti.to_function('f', [{x0, u_free, xk}, ps], {u_free, x_return});
                    end
                else
                    if obj.use_lambda
                        obj.f = obj.opti.to_function('f', ...
                            [{x0, u_free}, ps, {obj.opti.lam_g}], ...
                            {u_free, x_return, obj.opti.lam_g});
                    else
                        obj.f = obj.opti.to_function('f', [{x0, u_free}, ps], {u_free, x_return});
                    end
                end
            end

            obj.casadi_vars.x0 = x0;
            obj.casadi_vars.u_free = u_free;
            obj.casadi_vars.x1_N = x(:, 2:end);
            obj.casadi_vars.xk = xk;
            obj.casadi_vars.params = params;

            obj.free_inputs = mpcproblem.free_inputs;
            obj.fixed_inputs = mpcproblem.fixed_inputs;
            obj.parameters = mpcproblem.parameters;

            obj.mpcproblem = mpcproblem;
        end

        function step_result = solve_step(obj, step_input)
            
            x0 = step_input.x0;
            u_init = step_input.u_init;

            step_result = struct('x0', x0(:, 1));

            if obj.scaled
                x0 = scale_data(obj.scalings, obj.x_names, x0);
                u_init = scale_data(obj.scalings, get_signals(obj.free_inputs), u_init);
            end

            if isfield(step_input, 'lam_g')
                lam_g = step_input.lam_g;
            else
                lam_g = [];
            end

            if ~obj.use_lambda && ~isempty(lam_g)
                error("lam_g must be empty");
            elseif obj.use_lambda && isempty(lam_g)
                lam_g = zeros(size(obj.opti.lam_g));
            end

            param_values = cell(length(obj.parameters));

            for i = 1:length(obj.parameters)
                p = obj.parameters(i);
                pname = p.signal;

                if ~isfield(step_input, pname)
                    error("parameter '%s' not part of the step_input struct", pname);
                end

                value = step_input.(pname);

                if ~p.is_vector && (length(value) ~= 1)
                    error("parameter '%s' must be a scalar", pname);
                elseif p.is_vector
                    if length(value) == 1
                        value = value * ones(obj.N + 1, 1);
                    elseif length(value) ~= obj.N + 1
                        error("parameter '%s' must be a %d-dimensional (N+1-dimensional) vector", ...
                            pname, obj.N + 1);
                    end
                end

                param_values{i} = value;
            end

            keys = fieldnames(step_input);
            keys = setdiff(keys, get_signals(obj.parameters));
            keys = setdiff(keys, {'x0', 'u_init', 'lam_g'});

            if ~isempty(keys)
                error("unknown fields of step_input: %s", strjoin(keys, ', '));
            end

            init_args = {u_init};

            if obj.multiple_shooting
                if size(x0, 2) == 1
                    x_init = repmat(x0, 1, obj.N);
                else
                    x_init = x0(:, 2:end);
                    x0 = x0(:, 1);
                end

                init_args{end + 1} = x_init;
            else
                x0 = x0(:, 1);
            end

            if obj.use_opti_function
                if obj.use_lambda
                    [u_opt, x_opt, lam_g] = obj.f(x0, init_args{:}, param_values{:}, lam_g);
    
                    u_opt = full(u_opt);
                    x_opt = full(x_opt);
                else
                    [u_opt, x_opt] = obj.f(x0, init_args{:}, param_values{:});
    
                    u_opt = full(u_opt);
                    x_opt = full(x_opt);
                end
            else
                obj.opti.set_value(obj.casadi_vars.x0, x0);

                for i = 1:length(obj.parameters)
                    pname = obj.parameters(i).signal;
                    obj.opti.set_value(obj.casadi_vars.params.(pname), param_values{i});
                end
    
                obj.opti.set_initial(obj.casadi_vars.u_free, u_init);
    
                if obj.multiple_shooting
                    obj.opti.set_initial([obj.casadi_vars.xk], x_init);
                end
            
                if obj.use_lambda
                    obj.opti.set_initial(obj.opti.lam_g, lam_g);
                end
    
                sol = obj.opti.solve_limited();

                u_opt = full(sol.value(obj.casadi_vars.u_free));
                x_opt = full(sol.value([obj.casadi_vars.x1_N]));

                if obj.use_lambda
                    lam_g = sol.value(obj.opti.lam_g);
                end

                step_result.opti_sol = sol;
            end

            if obj.scaled
                step_result.u_opt = unscale_data(obj.scalings, get_signals(obj.free_inputs), u_opt);
                step_result.x_opt = unscale_data(obj.scalings, obj.x_names, x_opt);
            else
                step_result.u_opt = u_opt;
                step_result.x_opt = x_opt;
            end

            if obj.use_lambda
                step_result.lam_g = lam_g;
            else
                step_result.lam_g = [];
            end

            step_result.u_fixed = get_u_fixed(obj.fixed_inputs, obj.parameters, param_values, obj.N);
        end

        function u = get_complete_input(obj, u_free, u_fixed, oversample_rate)
            n = size(u_free, 2);

            u = zeros(length(obj.u_names), n * oversample_rate);

            constant_fixed = (size(u_fixed, 2) == 1);

            indices = get_input_indices(obj.u_names, obj.free_inputs, obj.fixed_inputs);

            idx1 = 0;
            for k = 1:n
                idx0 = idx1 + 1;
                idx1 = idx0 + oversample_rate - 1;

                for i = 1:length(indices)
                    if strcmp(indices(i).kind, 'free')
                        u(i, idx0:idx1) = u_free(indices(i).index, k);
                    else
                        if constant_fixed
                            u(i, idx0:idx1) = u_fixed(indices(i).index);
                        else
                            u(i, idx0:idx1) = u_fixed(indices(i).index, k);
                        end
                    end
                end
            end
        end

        function u_0 = get_u_init(obj)
            nu_free = length(obj.free_inputs);

            u_0 = zeros(nu_free, obj.N);
            
            for i = 1:nu_free
                u_0(i, :) = obj.free_inputs(i).init;
            end            
        end

        function [x_init, u_init, lam_g] = get_next_initial_values(obj, step_result)

            u_init = [step_result.u_opt(:, 2:end), step_result.u_opt(:, end)];
        
            if obj.multiple_shooting
                x_init = [step_result.x_opt, step_result.x_opt(:, end)];
            else
                x_init = step_result.x_opt(:, 1);
            end
        
            lam_g = step_result.lam_g;
        
        end

        function value = get_signal_value(obj, name, k, x, u)
            idx = find_element(obj.x_names, name);

            if (idx ~= 0)
                if isempty(x)
                    error("'%s' is a state, but no state values are provided", name);
                end

                value = x(idx, k);
                return;
            end

            idx = find_element(obj.u_names, name);
            if (idx ~= 0)
                if isempty(x)
                    error("'%s' is an input, but no input values are provided", name);
                end

                value = u(idx, k);
                return;
            end

            error("not implemented");
        end

        function cons = get_all_constraints(obj)
            cons = obj.mpcproblem.get_all_constraints();
        end

        function res = sol_into_simresult(obj, mpc_sol)
            u = obj.get_complete_input(mpc_sol.u_opt, mpc_sol.u_fixed, 1);
            t = (0:obj.N) * obj.Ts;
            x = [mpc_sol.x0, mpc_sol.x_opt];
            u(:, end + 1) = nan;

            f_out = casadi.Function('outputs', {obj.model.states, obj.model.inputs}, {obj.model.y});

            if obj.scaled
                x_sim = scale_data(obj.scalings, obj.x_names, x);
                u_sim = scale_data(obj.scalings, obj.u_names, u);
            else
                x_sim = x;
                u_sim = u;
            end

            y = zeros(length(obj.y_names), obj.N + 1);
            for i = 1:obj.N + 1
                if i == 1
                    ui = u_sim(:, 1);
                else
                    ui = u_sim(:, i - 1);
                end

                y(:, i) = full(f_out(x_sim(:, i), ui));
            end

            y_ft = zeros(length(obj.y_names), obj.N + 1);

            for i = 1:obj.N + 1
                y_ft(:, i) = full(f_out(x_sim(:, i), u_sim(:, i)));
            end

            if obj.scaled
                y = unscale_data(obj.scalings, obj.y_names, y);
                y_ft = unscale_data(obj.scalings, obj.y_names, y_ft);
            end
            
            res.t = t;
            res.u = u;
            res.x = x;
            res.y = y;
            res.y_ft = y_ft;
            res.u_names = obj.u_names;
            res.x_names = obj.x_names;
            res.y_names = obj.y_names;

            res.constraints = obj.get_all_constraints();
            
            res.meta.discrete = true;
            res.meta.Ts = obj.Ts;
            res.desc = '';            

            res.scaled = obj.orig_scaled;
            res.scalings = obj.model.scalings;
        end        
    end
end
