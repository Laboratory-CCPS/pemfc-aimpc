%%

clear("all"); %#ok<CLALL>
init_paths();

%%
clc();

p = params_suh();
model = get_model(@model_suh, p);


%% Setup MPC problem
% (used in all examples that follow)

mpcproblem = MpcProblem(model);

mpcproblem = mpcproblem.add_free_input('I_st', 0, inf, 130);
mpcproblem = mpcproblem.add_free_input('v_cm', 50, 250, 100);

mpcproblem = mpcproblem.add_constraint('p_O2', 0, inf);
mpcproblem = mpcproblem.add_constraint('p_N2', 0, inf);
mpcproblem = mpcproblem.add_constraint('p_sm', 0, inf);
mpcproblem = mpcproblem.add_constraint('w_cp', rpm2rad(20e3), rpm2rad(100e3));
mpcproblem = mpcproblem.add_constraint('lambda_O2', 1.5, 5, 'feedthrough', false);
mpcproblem = mpcproblem.add_constraint('lambda_O2', 1.5, 5, 'feedthrough', true);

mpcproblem = mpcproblem.add_quadratic_cost('I_st', '#I_load[]', 'weight_stage', 5e-2);
mpcproblem = mpcproblem.add_quadratic_cost('lambda_O2', 2, 'weight_stage', 1, 'weight_end', 1);

mpcproblem = mpcproblem.add_quadratic_diffquot_cost('v_cm', [-1, 1], 'ts_order', 0, 'weight', 1e-5);


%% Parameters for MPC solver
% (used in all examples that follow)

Ts_mpc = 0.025;
Npredict = 10;
discretization = 'rk1';
substeps = 2;
scale_model = true;
use_lambda = true;
multiple_shooting = true;

% options forwarded to casadi
plugin_options = struct();
solver_options = struct();
% solver_options = struct('max_iter', 100, 'dual_inf_tol', 1e-6, 'compl_inf_tol', 1e-6);


%% Example 1: Solve the optimization problem for one step

x0 = [0.1113e5; 0.8350e5; 6049.9; 1.5356e5]; % [p_O2, p_N2, w_cp, p_sm], for 130 A

I_load = @(time) 130 + (time >=  0.03) * 100;

mpcsolver = MpcSolver(mpcproblem, ...
    Ts_mpc, Npredict, discretization, 'substeps', substeps, ...
    'scale_model', scale_model, 'use_lambda', use_lambda, 'multiple_shooting', multiple_shooting, ...
    'solver_options', solver_options, 'plugin_options', plugin_options, ...
    'verbose', true);

% for parameters that are defined as vectors in the mpc problem setup the
% provided value must be either a scalar (which is then used for every time
% point) or as a vector of length N + 1, corresponding to the time points
% k = 0, ..., N, even if not all time points are used.
pref_I_st = I_load((0:Npredict).' * Ts_mpc);

u_init = mpcsolver.get_u_init();

step_input = struct('x0', x0, 'u_init', u_init, 'lam_g', [], 'I_load', pref_I_st);
step_result = mpcsolver.solve_step(step_input);

mpc_result = mpcsolver.sol_into_simresult(step_result);

plot_sim_result(mpc_result, 'SignalInfos', get_signal_infos_suh());


%% Example 2: Simulate the MPC with ideal model
% (That is, use the solution x(k + 1) of the optimization problem as
% state of the next step.)

% If future_I_load_known is true, the MPC is provided with the demanded
% current of each of the N time points.
future_I_load_known = false;

sim_oversampling = 1;
Ts_sim = Ts_mpc / sim_oversampling;

[u_ref, x0] = get_validation_input_suh(Ts_mpc);
I_load = @(t) u_ref(2, min(round(t / Ts_mpc) + 1, size(u_ref, 2)));

T_end = 29;

mpcsolver = MpcSolver(mpcproblem, ...
    Ts_mpc, Npredict, discretization, 'substeps', substeps, ...
    'scale_model', scale_model, 'use_lambda', use_lambda, 'multiple_shooting', multiple_shooting, ...
    'solver_options', solver_options, 'plugin_options', plugin_options, ...
    'verbose', false);

u_init = mpcsolver.get_u_init();
x_init = x0;
lam_g = [];

n_mpc_steps = floor(T_end / Ts_mpc);

u_mpc = nan(length(mpcsolver.free_inputs), n_mpc_steps);
u_fixed = nan(length(mpcsolver.fixed_inputs), n_mpc_steps);

pb = ProgressBar(n_mpc_steps);

for i = 1:n_mpc_steps
    pb = pb.updateinc();

    if future_I_load_known
        t_step = ((i - 1) + (0:mpcsolver.N).') * mpcsolver.Ts;
        I_load_step = I_load(t_step);
    else
        t_step = (i - 1) * mpcsolver.Ts;
        I_load_step = I_load(t_step);
    end

    step_input = struct('x0', x_init, 'u_init', u_init, 'lam_g', lam_g, 'I_load', I_load_step);
    step_result = mpcsolver.solve_step(step_input);
    
    [x_init, u_init, lam_g] = mpcsolver.get_next_initial_values(step_result);

    u_mpc(:, i) = step_result.u_opt(:, 1);
    u_fixed(:, i) = step_result.u_fixed(:, 1);
end

pb.delete();

u_sim = mpcsolver.get_complete_input(u_mpc, u_fixed, sim_oversampling);
idealmpc_res = sim_nl(model, Ts_sim, x0, u_sim);
idealmpc_res.constraints = mpcsolver.get_all_constraints();

plot_sim_result(idealmpc_res, 'SignalInfos', get_signal_infos_suh());


%% Example 3: Simulate the MPC with separately simulated model

future_I_load_known = false;

% Define true plant for simulation
% If the MPC ist implemented using "rk1" and substeps=4, the following
% should be the simulation model to have a perfect match:
%
% sim_oversampling = 4;
% Ts_sim = Ts_mpc / sim_oversampling;
% sim_model = discretize_model(model, Ts_sim, "rk1");


% 10 times oversampling (i.e. 10 simulation steps to simulate one mpc step)
% same modelo, but using the "idas" solver
%
% sim_oversampling = 10;
% Ts_sim = Ts_mpc / sim_oversampling;
% sim_model = model;

% 10 times oversampling, "idas" solver, parameter mismatch
%
sim_oversampling = 10;
Ts_sim = Ts_mpc / sim_oversampling;
p_sim = p;
p_sim.J_cp = p.J_cp * 0.5;
sim_model = get_model(@model_suh, p_sim);

simulator = Simulator(sim_model, Ts_sim);

T_end = 29;

mpcsolver = MpcSolver(mpcproblem, ...
    Ts_mpc, Npredict, discretization, 'substeps', substeps, ...
    'scale_model', scale_model, 'use_lambda', use_lambda, 'multiple_shooting', multiple_shooting, ...
    'solver_options', solver_options, 'plugin_options', plugin_options, ...
    'verbose', false);

[u_ref, x0] = get_validation_input_suh(Ts_mpc);
I_load = @(t) u_ref(2, min(round(t / Ts_mpc) + 1, size(u_ref, 2)));

u_init = mpcsolver.get_u_init();
x_init = x0;
lam_g = [];

n_mpc_steps = floor(T_end / Ts_mpc);

u_mpc = nan(length(mpcsolver.free_inputs), n_mpc_steps);
u_fixed = nan(length(mpcsolver.fixed_inputs), n_mpc_steps);

n_sim_steps = sim_oversampling * n_mpc_steps;
t_sim = (0:n_sim_steps) * Ts_sim;
x_sim = nan(length(sim_model.states), n_sim_steps + 1);
u_sim = nan(length(sim_model.inputs), n_sim_steps + 1);
y_sim = nan(length(sim_model.y_names), n_sim_steps + 1);
y_ft_sim = nan(length(sim_model.y_names), n_sim_steps + 1);

x_sim(:, 1) = x0;

pb = ProgressBar(n_mpc_steps);

for i = 1:n_mpc_steps
    pb = pb.updateinc();

    if future_I_load_known
        t_step = ((i - 1) + (0:mpcsolver.N).') * mpcsolver.Ts;
        I_load_step = I_load(t_step);
    else
        t_step = (i - 1) * mpcsolver.Ts;
        I_load_step = I_load(t_step(1));
    end

    step_input = struct('x0', x_init, 'u_init', u_init, 'lam_g', lam_g, ...
        'I_load', I_load_step);
    step_result = mpcsolver.solve_step(step_input);
    
    % shift the states by one step
    % (The first state will be overwritten below by the result of the
    % simulation.)
    [x_init, u_init, lam_g] = mpcsolver.get_next_initial_values(step_result);

    u_mpc(:, i) = step_result.u_opt(:, 1);
    u_fixed(:, i) = step_result.u_fixed(:, 1);

    % simulate plant
    sim_idx = sim_oversampling * (i - 1) + 1;
    u_sim_cur = mpcsolver.get_complete_input(u_mpc(:, i), u_fixed(:, i), 1);
    x_sim_cur = x_sim(:, sim_idx);

    for ii = 1:sim_oversampling 
        [x_sim_next, y_sim_next, y_sim_cur_ft] = simulator.sim_step(x_sim_cur, u_sim_cur);

        u_sim(:, sim_idx) = u_sim_cur;
        y_ft_sim(:, sim_idx) = y_sim_cur_ft;        

        x_sim(:, sim_idx + 1) = x_sim_next;
        y_sim(:, sim_idx + 1) = y_sim_next;

        x_sim_cur = x_sim_next;

        sim_idx = sim_idx + 1;
    end

    % use plant state as "true" state, i.e. use this as the initial state
    % for the next mpc step
    x_init(:, 1) = x_sim_next;
end

pb.delete();

res = build_sim_result_from_data(sim_model, t_sim, u_sim, x_sim, y_sim, y_ft_sim);
res.constraints = mpcsolver.get_all_constraints();

%plot_sim_result(res, 'SignalInfos', get_signal_infos_suh());
plot_sim_result({idealmpc_res, res}, 'SignalInfos', get_signal_infos_suh());
