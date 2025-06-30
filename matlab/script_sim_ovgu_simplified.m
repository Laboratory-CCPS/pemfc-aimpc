%%
clc();
init_paths();

%%

p = params_ovgu_simplified();
model = get_model(@model_ovgu_simplified, p);

%%

Ts = 0.01;
T_sim = 10;

n_sim = round(T_sim / Ts);

t = (0:n_sim - 1) * Ts;

data = load('testdata_ovgu_simplified.mat');
uInterpolant_pp = griddedInterpolant(data.u_traj_pp.time - data.u_traj_pp.time(1), data.u_traj_pp.data, 'pchip', 'nearest');

u = uInterpolant_pp(t).';

% u(:, 2:end) = u(:, 1) * ones(1, size(u, 2) - 1);

x0 = data.x0;

%%

res = sim_nl(model, Ts, x0, u);
res.desc = sprintf('continuous, Ts=%g s', Ts);
plot_sim_result(res, {'res: U_S, @inputs'});


%%

n_step = round(length(res.t) / 100);

idx = (1:n_step:length(res.t) - 1);
evs = nan(5, length(idx));
t_evs = res.t(idx);

for i = 1:length(idx)
    [A, B, C, D] = get_linearized_matrices(model, res.x(:, idx(i)), u(:, idx(i)));

    evs(:, i) = eig(A);
end

%%

vh = subplot(2, 1, 1);
plot(t_evs, real(evs).');
ylabel('real(ev)');

vh(2) = subplot(2, 1, 2);
plot(t_evs, 1 ./ abs(real(evs).'));
ylabel('1 / abs(ev)');
xlabel('time');
linkaxes(vh, 'x');

% disp('Eigenwerte von A:')
% disp(eig(A));
% 
% disp('1 / |lambda(A)|:')
% disp(1 ./ abs(eig(A)));
%%
subplot(5, 1, 1);
plot(t_evs, real(evs(1, :)).');
subplot(5, 1, 2);
plot(t_evs, real(evs(2, :)).');
subplot(5, 1, 3);
plot(t_evs, real(evs(3, :)).');
subplot(5, 1, 4);
plot(t_evs, real(evs(4, :)).');
subplot(5, 1, 5);
plot(t_evs, real(evs(5, :)).');
ylabel('real(ev)');
