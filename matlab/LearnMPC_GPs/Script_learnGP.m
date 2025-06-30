%% Script_learnGP
clc;clear all,close all
init_paths()



%% setup MPC Problem
p = params_suh();
model = get_model(@model_suh, p);


mpcproblem = MpcProblem(model);

mpcproblem = mpcproblem.add_free_input('I_st', 0, inf, 130);
mpcproblem = mpcproblem.add_free_input('v_cm', 50, 250, 200);

mpcproblem = mpcproblem.add_constraint('p_O2', 0, inf);
mpcproblem = mpcproblem.add_constraint('p_N2', 0, inf);
mpcproblem = mpcproblem.add_constraint('p_sm', 0, inf);
mpcproblem = mpcproblem.add_constraint('w_cp', rpm2rad(20e3), rpm2rad(105e3));
mpcproblem = mpcproblem.add_constraint('lambda_O2', 1.5, 5, 'feedthrough', false);
mpcproblem = mpcproblem.add_constraint('lambda_O2', 1.5, 5, 'feedthrough', true);

mpcproblem = mpcproblem.add_quadratic_cost('I_st', '#I_load[]', 'weight_stage', 5e-2);
mpcproblem = mpcproblem.add_quadratic_cost('lambda_O2', 2, 'weight_stage', 1, 'weight_end', 1);

mpcproblem = mpcproblem.add_quadratic_diffquot_cost('v_cm', [-1, 1], 'ts_order', 0, 'weight', 1e-5);
%mpctask.use_opti_function = false;

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
%% Mpc solver
mpcsolver = MpcSolver(mpcproblem, ...
    Ts_mpc, Npredict, discretization, 'substeps', substeps, ...
    'scale_model', scale_model, 'use_lambda', use_lambda, 'multiple_shooting', multiple_shooting, ...
    'solver_options', solver_options, 'plugin_options', plugin_options, ...
    'verbose', false);
%% collect MPC Data


X_range = [0.08e5 0.25e5;... %p_O2
    0.65e5 1.75e5; ... %p N2
    rpm2rad(30e3) rpm2rad(100e3);...%rpm2rad(105e3);...%w_cp
    1.3e5    2.7e5... % p_sm
    ];
Ref_range = [80 340];
num_samples_per_dim = 20000;

%Data = get_data(mpcsolver,X_range,Ref_range,num_samples_per_dim);
percentages.new_sample = 0.2;
percentages.closed_loop_noise = 0.3;
percentages.closed_loop = 0.5;
num_data = 20000;% 10000;
Data = get_data_closed_loop(mpcsolver,X_range,Ref_range,percentages,num_data);


% or load collected data

%load('DataFCMPC.mat')

% % remove NaN from data
Data_noNaN = remove_row_with_nancolumn(Data,3);

which_u = 1;
Data_GP_u1 = data_to_GP_format(Data_noNaN,which_u);
% u2
which_u = 2;
Data_GP_u2 = data_to_GP_format(Data_noNaN,which_u);




%transform
%[Data_GP_u1.X,info_trans_X_u1] = transform_data(Data_GP_u1.X,'rescale');
[Data_GP_u1.X,info_trans_X_u1] = transform_data(Data_GP_u1.X,'zscore');
[Data_GP_u1.Y,info_trans_Y_u1] = transform_data(Data_GP_u1.Y,'zscore');


%transform
%[Data_GP_u2.X,info_trans_X_u2] = transform_data(Data_GP_u2.X,'rescale');
[Data_GP_u2.X,info_trans_X_u2] = transform_data(Data_GP_u2.X,'zscore');
[Data_GP_u2.Y,info_trans_Y_u2] = transform_data(Data_GP_u2.Y,'zscore');

%% Design GP
run('gpml-matlab-v4.2-2018-06-11/gpml-matlab-v4.2-2018-06-11/startup.m')
load_gp = true;% or "false". if "false" the GP is designed anew. if "true" an existing GP model can be loaded

if ~load_gp
    %GP_settings.kernelfunc = {@covSEard};
    GP_settings.kernelfunc = {@covNNone}; 
    %GP_settings.kernelfunc = {'covSum',{@covSEard, @covNNone}};
    %GP_settings.kernelfunc = {@covNNard_gpt};
    GP_settings.meanfunc = @meanZero;
    GP_settings.likfunc = @likGauss;
    ell_1=1;
    ell_2=1;
    ell_3=1;
    ell_4=1;
    ell_5=1;%100;
    sf2 = 1;
    s_noise2 = 0.001;
    GP_settings.s_noise2 = s_noise2;
    %GP_settings.log_Hyperpara_init = log([ell_1 ell_2 ell_3 ell_4 ell_5 sf2]); % SEard
    %GP_settings.log_Hyperpara_init = log([ell_1 ell_2 ell_3 ell_4 sf2]); % SEard const ref
    GP_settings.log_Hyperpara_init = log([ell_1 sf2]); % NNone

    %%%%%%%%%% first input
    GP_u1 = MyGP(Data_GP_u1,GP_settings,info_trans_X_u1,info_trans_Y_u1);
    GP_u1.fixed_noise = true;
    % GP_u1.log_hyp_opt_struct.mean = [];
     %GP_u1.log_hyp_opt_struct.cov = [0.2713 -1.5263 0.1232 -1.7675 0.1863 0.7457 -3.5516 -1.7027];
    % GP_u1.log_hyp_opt_struct.lik = -6.9078;




    GP_u1 = GP_u1.hyperpara_opti(2000);

    batch_mode = 0; % if false we always choose among all data; if true we choose among a randomly sampled set (resampled each time we choose)
    batch_size = 100000;
    select_scoring_based = 1;
    additionally_select_random = 10000 - select_scoring_based;
    GP_u1 = GP_u1.choosePoints(select_scoring_based,batch_mode,batch_size,additionally_select_random);
    %%%%%%%%%% second input


    GP_u2 = MyGP(Data_GP_u2,GP_settings,info_trans_X_u2,info_trans_Y_u2);
    GP_u2.fixed_noise = true;
    GP_u2 = GP_u2.hyperpara_opti(2000); % 
    GP_u2 = GP_u2.choosePoints(select_scoring_based,batch_mode,batch_size,additionally_select_random);



    %% save and load created gps:
    %save
     % save('GP_u1_good.mat', 'GP_u1');
     % save('GP_u2_good.mat', 'GP_u2');
    
else
    % load

        loaded_gpObj = load('GP_u1_good.mat', 'GP_u1');
    GP_u1 = loaded_gpObj.GP_u1;

    loaded_gpObj = load('GP_u2_good.mat', 'GP_u2');
    GP_u2 = loaded_gpObj.GP_u2;


end
%% if you want set different hyperparameters (example)
    % GP_u1.log_hyp_opt_struct.mean = [];
    % GP_u1.log_hyp_opt_struct.cov = [0.2713 -1.5263 0.1232 -1.7675 0.1863 0.7457 -3.5516 -1.7027];
    % GP_u1.log_hyp_opt_struct.lik = -6.9078;
    %     GP_u2.log_hyp_opt_struct.mean = [];
    % GP_u2.log_hyp_opt_struct.cov = [0.9239 -0.1855 -1.1514 -0.4179 -0.7137 0.2948 -2.2882 0.2542];
    % GP_u2.log_hyp_opt_struct.lik = -6.9078;


%% if you want to "reselect" points in the GP (example)
% this takes quite a while for large  number of points as in 
%each step (num_point_steps) a nxn matrix is inverted, where n =
% %1...num_points increases with each step
% GP_u1 = GP_u1.reset_gp_data();
% GP_u1 = GP_u1.choosePoints(3000);
% GP_u2 =GP_u2.reset_gp_data();
% GP_u2 =GP_u2.choosePoints(3500);

%%

future_I_load_known = false;
use_lambda = true;

% Define true plant for simulation
% If the MPC is implemented using "rk1" and substeps=4, the following
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
%p_sim.J_cp = p.J_cp * 0.9; % parameter mismatch
sim_model = get_model(@model_suh, p_sim);

simulator = Simulator(sim_model, Ts_sim);

Np = mpcsolver.N;

[u_ref, x0] = get_validation_input_suh(Ts_mpc);

I_load = @(t) u_ref(2, min(round(t / Ts_mpc) + 1, size(u_ref, 2)));

T_end = 29;

n_mpc_steps = floor(T_end / Ts_mpc);

u_init = mpcsolver.get_u_init();
x_init = x0;
x_init_gp = x_init;
lam_g = [];


steps = 1160;%1160;%100;%1160;

[t_sim, u_sim, x_sim, y_sim, y_ft_sim,t_comp_mpc] = simulator.sim_closed_loop(mpcsolver,'mpc',Ts_mpc,x0,I_load,steps);



infoX_GP1 = GP_u1.data_transform_info_X;
infoY_GP1 = GP_u1.data_transform_info_Y;
infoX_GP2 = GP_u2.data_transform_info_X;
infoY_GP2 = GP_u2.data_transform_info_Y;

% Gp controller function handle
controller = @(x,ref) [detransform_data(GP_u1.predict_fast_ish(  transform_data([x' ref],'',infoX_GP1)  ),infoY_GP1); detransform_data(GP_u2.predict_fast_ish(  transform_data([x' ref],'',infoX_GP2)  ),infoY_GP2)];

% timeit(@()controller(x,ref)) % estimated computation time
[t_sim_gp, u_sim_gp, x_sim_gp, y_sim_gp, y_ft_sim_gp,t_comp_gp] = simulator.sim_closed_loop(controller,'gp',Ts_mpc,x0,I_load,steps);


res = build_sim_result_from_data(sim_model, t_sim, u_sim, x_sim, y_sim, y_ft_sim);
res.desc = 'MPC';
res_gp = build_sim_result_from_data(sim_model, t_sim_gp, u_sim_gp, x_sim_gp, y_sim_gp, y_ft_sim_gp);
res_gp.desc = 'GP';

res.constraints = mpcsolver.get_all_constraints();
res_gp.constraints = mpcsolver.get_all_constraints();



plot_sim_result({res,res_gp}, 'SignalInfos', get_signal_infos_suh());%,'ReuseFigures', false);


figure()
hold on
plot(t_comp_mpc,'kx')
plot(t_comp_gp,'ro');
legend('MPC','GP')
xlabel('iteration')
ylabel('comp time')


GP_is_this_much_faster = mean(t_comp_mpc)/mean(t_comp_gp)

% mpc time (milliseconds)
t_comp_mpc(1) =[]; % this is mpc start up, often an outlier
mean(t_comp_mpc)*10^3
median(t_comp_mpc)*10^3
std(t_comp_mpc.*10^3)
min(t_comp_mpc)*10^3
max(t_comp_mpc)*10^3
%gp
mean(t_comp_gp)*10^3
median(t_comp_gp)*10^3
std(t_comp_gp.*10^3)
min(t_comp_gp)*10^3
max(t_comp_gp)*10^3
figure()
hold on
subplot(121)
plot(res.t,res.u(1,:),'b-')
hold on
plot(res_gp.t,res_gp.u(1,:),'r-')
ylim([50 300])
subplot(122)
plot(res.t,res.u(2,:),'b-')
hold on
plot(res_gp.t,res_gp.u(2,:),'r-')
ylim([0 400])

%% states
figure()
hold on
subplot(2,1,1)
hold on
plot(res.t,res.x(1,:).*10^-5,'Color',[0.8500 0.3250 0.0980])
plot(res.t,res.x(2,:).*10^-5,'Color',[0.4660 0.6740 0.1880])
plot(res.t,res.x(4,:).*10^-5,'Color',[0 0.4470 0.7410])
plot(res.t,res_gp.x(1,:).*10^-5,'k--')
plot(res.t,res_gp.x(2,:).*10^-5,'k--')
plot(res.t,res_gp.x(4,:).*10^-5,'k--')
grid on
legend('o2 mpc','n2 mpc','sm mpc','o2 gp','n2 gp','sm gp')
xlabel('time s')
ylabel('pressure bar')

subplot(2,1,2)
hold on
plot(res.t,rad2rpm(res_gp.x(3,:)).*10^-3,'k--')
plot(res.t,rad2rpm(res.x(3,:)).*10^-3,'Color',[0.8500 0.3250 0.0980])
%line([0 res.t(end)],[20 20],'Color','red','LineStyle','--')
line([0 res.t(end)],[105 105],'Color','red','LineStyle','--')
grid on
xlabel('time s')
ylabel('motor vel krpm')
% plot line 20 to 105 krpm
% cleanfigure()
% matlab2tikz('FCstates_2.tex')
%% controls
figure()
hold on
subplot(2,1,1)
grid on
hold on
plot(res.t,res.u(1,:),'Color',[0.8500 0.3250 0.0980])
plot(res.t,res_gp.u(1,:),'k--')
xlabel('t s')
ylabel('motor voltage')
line([0 res.t(end)],[50 50],'Color','red','LineStyle','--')
line([0 res.t(end)],[250 250],'Color','red','LineStyle','--')
subplot(2,1,2)
grid on
hold on
plot(res.t,res.u(2,:),'Color',[0.8500 0.3250 0.0980])
plot(res.t,res_gp.u(2,:),'k--')
xlabel('t s')
ylabel('stack current')
% cleanfigure()
% matlab2tikz('FCcontrols2.tex')
%% oxygen excess
figure()
grid on
hold on
plot(res.t,res.y_ft(1,:),'Color',[0.8500 0.3250 0.0980])
hold on
plot(res.t,res_gp.y_ft(1,:),'k--')
xlabel('t s')
ylabel('lambda O2 -')
line([0 res.t(end)],[1.5 1.5],'Color','red','LineStyle','--')
% cleanfigure()
% matlab2tikz('FCoxygen.tex')
%%

endofscript = 1;