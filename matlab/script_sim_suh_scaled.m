%%
clc();
init_paths();

%%
p = params_suh();
model = get_model(@model_suh, p);

smodel = scale_model(model);
usmodel = unscale_model(smodel);

%%

Ts = 0.01;
[u, x0] = get_validation_input_suh(Ts);

res = sim_nl(model, Ts, x0, u);
res.desc = 'original';

usres = sim_nl(usmodel, Ts, x0, u);
usres.desc = 'unscale(scale(model))';

us = scale_model_signals(model, 'u', u);
x0s = scale_model_signals(model, 'x', x0);

sres = sim_nl(smodel, Ts, x0s, us);
sres.desc = 'scale(model)';

sresu = unscale_sim_results(sres);
sresu.desc = 'scale(model) -> unscale';

plot_sim_result({res, usres, sresu}, 'SignalInfos', get_signal_infos_suh());
plot_sim_result(sres, 'SignalInfos', get_signal_infos_suh(), 'AutoUnscale', false);
