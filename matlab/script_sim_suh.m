%%
clc();
init_paths();

%%
p = params_suh();
model = get_model(@model_suh, p);

%%

Ts = 0.01;
[u, x0] = get_validation_input_suh(Ts);

res = sim_nl(model, Ts, x0, u);
res.desc = sprintf('continuous, Ts=%g s', Ts);
plot_sim_result(res, 'SignalInfos', get_signal_infos_suh());

%%
[A, B, C, D] = get_linearized_matrices(model, x0, u(:, 1));

disp('Eigenwerte von A:')
disp(eig(A));

disp('1 / |lambda(A)|:')
disp(1 ./ abs(eig(A)));

%%

Ts = 0.001;
[u, x0] = get_validation_input_suh(Ts);

dmodel = discretize_model(model, Ts, 'rk4');
dres = sim_nl(dmodel, Ts, x0, u);
dres.desc = sprintf('discrete, Ts=%g s', Ts);
plot_sim_result(dres, 'SignalInfos', get_signal_infos_suh());

%%

plot_sim_result({res, dres}, 'ReuseFigure', true, 'SignalInfos', get_signal_infos_suh());


%%

n_step = round(length(res.t) / 100);

idx = (1:n_step:length(res.t) - 1);
evs = nan(length(model.states), length(idx));
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
