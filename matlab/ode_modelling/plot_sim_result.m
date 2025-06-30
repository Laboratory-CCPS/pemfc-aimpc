function plot_sim_result(res, varargin)

    ip = inputParser();
    ip.KeepUnmatched = false;

    ip.addOptional('PlotInfo', {': @states, @outputs, @inputs'});
    ip.addParameter('ReuseFigures', false, @islogical);
    ip.addParameter('SignalInfos', struct(), @isstruct);
    ip.addParameter('AutoUnscale', true, @islogical);

    ip.parse(varargin{:});

    reuseFigures = ip.Results.ReuseFigures;
    signal_infos = ip.Results.SignalInfos;
    plot_info = ip.Results.PlotInfo;
    auto_unscale = ip.Results.AutoUnscale;

    if ~iscell(res)
        res = {res};
    end

    nResults = length(res);

    x_names = {};
    y_names = {};
    u_names = {};

    scaled = res{1}.scaled;

    for i = 1:nResults
        [~, idx] = setdiff(res{i}.x_names, x_names, 'stable');
        x_names = [x_names, res{i}.x_names(idx).']; %#ok<AGROW>

        [~, idx] = setdiff(res{i}.y_names, y_names, 'stable');
        y_names = [y_names, res{i}.y_names(idx).']; %#ok<AGROW>

        [~, idx] = setdiff(res{i}.u_names, u_names, 'stable');
        u_names = [u_names, res{i}.u_names(idx).']; %#ok<AGROW>

        if res{i}.scaled
            if auto_unscale
                res{i} = unscale_sim_results(res{i});
            elseif res{i}.scaled ~= scaled
                error("the results must be all scaled or all unscaled");
            end
        end
    end

    if auto_unscale
        scaled = false;
    end

    if nResults == 1 && isempty(res{1}.desc)
        legendtexts = {};
    else
        legendtexts = cell(1, length(res));

        for r = 1:nResults
            if isempty(res{r}.desc)
                legendtexts{r} = sprintf("sim. %d", r);
            else
                legendtexts{r} = res{r}.desc;
            end
        end
    end


    for ifig = 1:length(plot_info)
        [title_str, names] = parse_plot_info(plot_info{ifig});

        names = replace_cat_names(names, x_names, u_names, y_names);

        if isempty(title_str)
            if reuseFigures
                clf();
            else
                figure();
            end
        elseif reuseFigures
            figureX(title_str);
            clf();
        else
            figure('Name', title_str);
        end

        % subfigure info
        nSubplots = length(names);

        if nSubplots < 5
            nCols = 1;
        else
            nCols = 2;
        end

        nRows = ceil(nSubplots / nCols);
        vh = [];

        for ip = 1:nSubplots
            if ip == 1
                vh = subplot(nRows, nCols, ip);
            else
                vh(end+1) = subplot(nRows, nCols, ip); %#ok<AGROW>
            end

            signal = names{ip};

            curlegendtexts = {};

            for ir = 1:nResults
                if ir == 1
                    basecolor = [0, 0, 1];
                elseif ir == 2
                    basecolor = [1, 0, 0];
                elseif ir == 3
                    basecolor = [0, 1, 0];
                elseif ir == 4
                    basecolor = [1, 0, 1];
                else
                    error('More than four results cannot be plotted together.')
                end

                plotted = plot_(vh(ip), res{ir}, signal, signal_infos, basecolor, scaled);
                hold(vh(ip), "on");

                if plotted && ~isempty(legendtexts)
                    curlegendtexts{end + 1} = legendtexts{ir}; %#ok<AGROW>
                end
            end
            hold(vh(ip), "off");

            if ~isempty(curlegendtexts)
                legend(curlegendtexts);
            end

            if (ip >= nSubplots - 1)
                if isfield(signal_infos, 'time')
                    info = signal_infos.time;
                    xlabel(vh(ip), sprintf('%s / %s', info.tex, info.disp_unit), "Interpreter", "tex");
                else
                    xlabel(vh(ip), 'time');
                end
            end

            if isfield(signal_infos, signal)
                info = signal_infos.(signal);

                sig_disp_string = info.tex;
                
                if scaled
                    sig_ylabel_string = sprintf("%s (scaled)", info.tex);
                elseif isempty(info.disp_unit)
                    sig_ylabel_string = info.tex;
                else
                    sig_ylabel_string = sprintf("%s / %s", info.tex, info.disp_unit);
                end
                    
                sig_disp_interp = 'tex';

            elseif ~isempty(signal_infos)
                sig_disp_string = signal;
                
                if scaled
                    sig_ylabel_string = sprintf("%s (scaled)", signal);
                else
                    sig_ylabel_string = signal;
                end
                
                sig_disp_interp = 'none';
            end

            title(vh(ip), sig_disp_string, 'Interpreter', sig_disp_interp);
            ylabel(vh(ip), sig_ylabel_string, 'Interpreter', sig_disp_interp);

            grid(vh(ip), 'on');
        end
    end

    if ~isempty(vh)
        linkaxes(vh, 'x');
    end    
end

function plotted = plot_(hax, res, signal, signal_infos, basecolor, scaled)

    plotted = true;
    stair_plot = false;

    [idx, type] = find_signal(res, signal);

    switch type
        case 'x'
            values = res.x(idx, :);
            t = res.t;

        case 'y'
            values = res.y(idx, :);
            t = res.t;
    
            if ismember('y_ft', fieldnames(res))
                values_ft = res.y_ft(idx, :);
    
                t = [t; t];
                values = [values; values_ft];
                t = t(2:end-1);
                values = values(2:end-1);
            end
        case 'u'
            stair_plot = true;
            values = res.u(idx, :);
            t = res.t;
        otherwise
            plotted = false;
            return;
    end

    if ~scaled && ismember(signal, fieldnames(signal_infos))
        info = signal_infos.(signal);
    else
        info = struct('tex', signal, 'unit', '', 'disp_unit', '', 'disp_fct', @(x) x);
    end

    if ismember('time', fieldnames(signal_infos))
        t_info = signal_infos.time;
    else
        t_info = struct('tex', 'time', 'unit', '', 'disp_unit', '', 'disp_fct', @(x) x);
    end
    
    if ~stair_plot
        plot(t_info.disp_fct(t), info.disp_fct(values), 'color', basecolor);
    else
        stairs(t_info.disp_fct(t), info.disp_fct(values), 'color', basecolor);
    end

    if ismember('constraints', fieldnames(res))
        constraints = res.constraints;

        if ismember(signal, fieldnames(constraints))
            sc = constraints.(signal);

            if sc.min > -inf
                h = yline(hax, ...
                            info.disp_fct(sc.min), ...
                            'LineStyle', ':', 'Color', 'k');
                excludeFromLegend(h);
            end
            if sc.max < inf
                h = yline(hax, info.disp_fct(sc.max), ...
                            'LineStyle', ':', 'Color', 'k');
                excludeFromLegend(h);
            end
        end
    end
end

function [title, signals] = parse_plot_info(plot_info_entry)

    title_signals = strsplit(plot_info_entry, ':');

    title = strtrim(title_signals{1});
    signals = strsplit(title_signals{2}, ',');

    signals = cellfun(@strtrim, signals, 'UniformOutput', false);

end


function names = replace_cat_names(names, x_names, u_names, y_names)

    [~, idx] = ismember('@states', names);
    if idx > 0
        names = [names(1:idx-1), x_names, names(idx+1:end)];
    end

    [~, idx] = ismember('@outputs', names);
    if idx > 0
        names = [names(1:idx-1), y_names, names(idx+1:end)];
    end

    [~, idx] = ismember('@inputs', names);
    if idx > 0
        names = [names(1:idx-1), u_names, names(idx+1:end)];
    end
end


function [idx, type] = find_signal(res, name)

    [~, idx] = ismember(name, res.x_names);
    if idx > 0
        type = 'x';
        return;
    end

    [~, idx] = ismember(name, res.y_names);
    if idx > 0
        type = 'y';
        return;
    end

    [~, idx] = ismember(name, res.u_names);
    if idx > 0
        type = 'u';
        return;
    end

    type = '';

end

