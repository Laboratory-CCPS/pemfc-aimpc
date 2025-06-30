classdef Simulator

    properties
        f_x
        F
        f_out
        Ts
        meta
        model
    end

    methods
        function obj = Simulator(model, Ts)
            obj.model = model;
            discrete = ismember('x_next', fieldnames(model));
            cont = ismember('dx', fieldnames(model));

            assert(xor(discrete, cont));

            obj.Ts = Ts;
            obj.meta.discrete = discrete;

            if discrete
                assert(Ts == model.Ts);

                obj.f_x = casadi.Function('f', {model.states, model.inputs}, {model.x_next});
                obj.F = [];
            else
                t0 = 0;
                tf = Ts;

                dae = struct('x', model.states, 'p', model.inputs, 'ode', model.dx);
                opts = struct('t0', t0, 'tf', tf);
                obj.F = casadi.integrator('F', 'idas',dae, opts);
                obj.f_x = [];
            end

            obj.f_out = casadi.Function('outputs', {model.states, model.inputs}, {model.y});
        end

        function [x, y, y0_ft] = sim_step(obj, x0, u)
            if obj.meta.discrete
                x = obj.f_x(x0, u);
            else
                r = obj.F('x0', x0, 'p', u);
                x = r.xf;
            end

            y = full(obj.f_out(x, u));
            x = full(x);

            if (nargout == 3)
                y0_ft = full(obj.f_out(x0, u));
            end
        end

        function y = eval_outputs(obj, x, u)
            y = full(obj.f_out(x, u));
        end

        function [t_sim, u_sim, x_sim, y_sim, y_ft_sim, t_cpu_controller] = sim_closed_loop(obj,controller,controller_type,Ts_controller,x0,Ref_load_traj,steps)
            % only constant ref load over prediction horizon allowed
            % TODO: think of a nice way to integrate the controller
            % function (remove distiction between MPC and learned controller etc...)
            % i first wanted controller to be a function handle like
            % controller = @(x,ref) mpctask.solve_step...; but because of
            % updating init guess  i have to think how to do it in a good
            % way

            % u_fixed? for appro

            
            Ts_sim = obj.Ts;%Ts_controller / sim_oversampling;
            sim_oversampling = Ts_controller/Ts_sim;
            %n_sim_steps = sim_oversampling * steps;
            x_init = x0;


            u_cont = nan(length(obj.model.inputs), steps); % mpctask.nu_free make it variable
            %u_fixed = nan(length(mpcsolver.fixed_inputs), steps);

            n_sim_steps = sim_oversampling * steps;

            t_sim = (0:n_sim_steps) * Ts_sim;
            x_sim = nan(length(obj.model.states), n_sim_steps + 1); % length(sim_model.states)
            u_sim = nan(length(obj.model.inputs), n_sim_steps + 1);
            y_sim = nan(length(obj.model.y_names), n_sim_steps + 1);
            y_ft_sim = nan(length(obj.model.y_names), n_sim_steps + 1);

            x_sim(:, 1) = x0;
            if strcmp(controller_type, 'mpc')
                u_fixed = nan(length(controller.fixed_inputs), steps);
                u_init = controller.get_u_init();
                lam_g = [];
            end
            pb = ProgressBar(steps);
            t_cpu_controller = nan(1,steps);
            for i = 1:steps
                %fprintf("step %d / %d\n", i, steps);
                pb = pb.updateinc();
                % get control input
                t_step = (i-1)*Ts_controller;
                Ref_load = Ref_load_traj(t_step);

                if strcmp(controller_type, 'mpc')
                    %u_i = controllerfun(x_init,Ref_load,u_init,lam_g);%,u_init,lam_g);
                    step_input = struct('x0', x_init, 'u_init', u_init, 'lam_g', lam_g, ...
                        'I_load', Ref_load);
                    %tStart = cputime;
                    tStart = tic;
                    step_result = controller.solve_step(step_input);
                    %tEnd = cputime - tStart;
                    tEnd = toc(tStart);
                    t_cpu_controller(i) = tEnd;
                    %t_cpu_controller(i) = timeit(@()controller.solve_step(step_input));% if you want to be accurate use timeit; but then the simulation takes long as timeit solves the ocp multiple times
                    [x_init, u_init, lam_g] = controller.get_next_initial_values(step_result);
                    % u_init(:, 1:controller.N-1) = step_result.u_opt(:, 2:controller.N);
                    % u_init(:, controller.N) = step_result.u_opt(:, controller.N);

                    % if controller.multiple_shooting
                    %     x_init = [step_result.x_opt, step_result.x_opt(:, end)];
                    % else
                    %     x_init = step_result.x_opt(:, 1);
                    % end

                    lam_g = step_result.lam_g;

                    u_i = step_result.u_opt(:, 1);
                    u_fixed(:, i) = step_result.u_fixed(:, 1);
                else 
                    %tStart = cputime; % using cputime results in tEnd=0 here..it is not accurate enough  https://stackoverflow.com/questions/56716774/matlab-cpu-time-0-issue#:~:text=The%20resolution%20of%20the%20clock,(Just%20In%20Time)%20compiler.
                    u_i = controller(x_init,Ref_load);
                    %tEnd = cputime - tStart;
                    %t_cpu_controller(i) = tEnd;
                    t_cpu_controller(i) = timeit(@()controller(x_init,Ref_load)); % timit is recommended

                end
                % Np = 10;
                % u_init(:, 1:Np-1) = step_result.u_opt(:, 2:Np);
                % u_init(:, Np) = step_result.u_opt(:, Np);
                u_cont(:, i) = u_i;
                % simulate plant
                sim_idx = sim_oversampling * (i - 1) + 1;
                if strcmp(controller_type, 'mpc')
                    u_sim_cur = controller.get_complete_input(u_i,u_fixed(:,i), 1);%;%u_i;%mpctask.get_complete_input(u_i, 1);
                else
                    u_sim_cur = u_i(end:-1:1);
                end
                x_sim_cur = x_sim(:, sim_idx);

                for ii = 1:sim_oversampling
                    [x_sim_next, y_sim_next, y_sim_cur_ft] = obj.sim_step(x_sim_cur, u_sim_cur);
                    u_sim(:, sim_idx) = u_sim_cur;
                    y_ft_sim(:, sim_idx) = y_sim_cur_ft;




                    x_sim(:, sim_idx + 1) = x_sim_next;
                    y_sim(:, sim_idx + 1) = y_sim_next;



                    x_sim_cur = x_sim_next;
                    sim_idx = sim_idx + 1;

                end % ii = 1:sim_oversampling
                x_init(:, 1) = x_sim_next;
            end % for i = 1:steps
            pb.delete();



        end % sim_closed_loop

    end
end
