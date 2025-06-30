function Data = get_data_closed_loop(mpctask,X_range,Ref_range,percentages,num_data)
%rng(25468,'twister') % for reproducability

% how to sample
percent_new = percentages.new_sample; %sample completely new
percent_cl = percentages.closed_loop; % new sample is solution from previous
%percentage_cl_noise = precentages.closed_loop_noise; % new sample is solution from previous but with some noise

dim_ref = 1; % dimension of reference
dim_x = length(X_range);


% for now just x_0, Iref Xocp, u_ocp
sample_points_X = [];
sample_points_Ref = [];

Data = {};
sample_ref = unifrnd(Ref_range(1,1), Ref_range(1,2), 1,1);
sample_x = unifrnd(X_range(:,1), X_range(:,2));
for k = 1:num_data
    %k




    trysolve = 1;
    already_newsample = false;
    %iter_try = 0;
    while trysolve
        %iter_try = iter_try + 1
        try
            mpctask.free_inputs(1).init = sample_ref;
            u_init = mpctask.get_u_init();
            step_input = struct('x0', sample_x, 'u_init', u_init, 'lam_g', [], 'I_load', sample_ref*ones(mpctask.N+1,1));
            step_result = mpctask.solve_step(step_input);
            trysolve = 0;
        catch
            sample_ref = unifrnd(Ref_range(1,1), Ref_range(1,2), 1,1);
            sample_x = unifrnd(X_range(:,1), X_range(:,2));
            already_newsample = true;

            %
            %step_result.x_opt = NaN;
            %step_result.u_opt = NaN;
            %step_result.sol = NaN;
        end
    end % while try new ic until feasible was found
    %sample_points_X(:,k)
    Data{k,1} = sample_x;
    Data{k,2} = sample_ref; % constant load over prediction horizon
    Data{k,3} = step_result.x_opt;
    Data{k,4} = step_result.u_opt;
    % Data{k,5} = J_ocp;
    % Data{k,6} = X_cl;
    % Data{k,7} = U_cl;
    % Data{k,8} = J_cl;

    % choose new sample
    % draw random number
    rnd = rand;
if ~already_newsample
    if rnd < percent_new
        % completely new (notice that we also sample completely new if
        % initial condition was infeasible
        sample_ref = unifrnd(Ref_range(1,1), Ref_range(1,2), 1,1);
        sample_x = unifrnd(X_range(:,1), X_range(:,2));

    elseif rnd < percent_new + percent_cl %
        % closed loop
        sample_ref = sample_ref;%
        sample_x = step_result.x_opt(:,1);

    else
        %closed loop plus noise
        sample_ref = sample_ref+5*randn;

        sample_x = step_result.x_opt(:,1) + 0.05.*randn(4,1).*(step_result.x_opt(:,1));

    end % if (how to sample)
end % if ~already_newsample only resample if we have not already resampled in the catch condition...

end % k = 1:num_data


end %fct