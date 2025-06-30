function info = get_signal_infos_suh(signal)

    if (nargin < 1), signal = ''; end

    infos.time = struct('tex', 'time', 'unit', 's', 'disp_unit', 's', 'disp_fct', @(x) x);

    infos.p_O2 = struct('tex', 'p_{O2}', 'unit', 'Pa', 'disp_unit', 'bar', 'disp_fct', @(x) x * 1e-5);
    infos.p_N2 = struct('tex', 'p_{N2}', 'unit', 'Pa', 'disp_unit', 'bar', 'disp_fct', @(x) x * 1e-5);
    infos.p_sm = struct('tex', 'p_{sm}', 'unit', 'Pa', 'disp_unit', 'bar', 'disp_fct', @(x) x * 1e-5);
    infos.w_cp = struct('tex', '\omega_{cp}', ...
        'unit', 'rad/s', 'disp_unit', 'krpm', 'disp_fct', @(x) rad2rpm(x) * 1e-3);
    infos.lambda_O2 = struct('tex', '\lambda_{O2}', 'unit', '1', 'disp_unit', '', 'disp_fct', @(x) x);
    infos.I_st = struct('tex', 'I_{st}', 'unit', 'A', 'disp_unit', 'A', 'disp_fct', @(x) x);
    infos.v_cm = struct('tex', 'v_{cm}', 'unit', 'V', 'disp_unit', 'V', 'disp_fct', @(x) x);
    
    if isempty(signal)
        info = infos;
    else
        info = infos.(signal);
    end
  
end
