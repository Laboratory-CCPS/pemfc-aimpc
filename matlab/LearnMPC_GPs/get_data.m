%% collect_data

%currently only captures infeasible points or other problems in the
%optimzation properly if mpctask.use_opti_function = false; otherwise you
%may store infeasible solutions.

%sample unifromly in box defined by
%X_range
function Data = get_data(mpctask,X_range,Ref_range,Num_samples_dim)
%Data = { x_0, (Iref),  x_ocp_opt, u_ocp_opt, J_ocp, x_cl, u_cl, J_cl}



%rng(1354,'twister') % for reproducability
dim_ref = 1; % dimension of reference
dim_x = length(X_range); 


% for now just x_0, Iref Xocp, u_ocp
sample_points_X = [];
sample_points_Ref = [];
for k = 1:dim_x
    sample_points_X(k,:) = unifrnd(X_range(k,1), X_range(k,2), 1,Num_samples_dim);
end
for k = 1:dim_ref
    sample_points_Ref(k,:) = unifrnd(Ref_range(k,1), Ref_range(k,2), 1,Num_samples_dim);
end


Data = {}; % x0, ref, Xsol, Usol, (Jsol, xcl,ucl,Jcl)
for k = 1:Num_samples_dim%*(dim_ref + dim_x)
    k
    mpctask.free_inputs(1).init = sample_points_Ref(:,k);
    u_init = mpctask.get_u_init();
    step_input = struct('x0', sample_points_X(:,k), 'u_init', u_init, 'lam_g', [], 'I_load', sample_points_Ref(:,k)*ones(mpctask.N+1,1));
    %step_input = struct('x0', x0, 'u_init', mpctask.get_u_init(), 'lam_g', [], 'I_load', sample_points_Ref(:,k)*ones(mpctask.N,1));
    try
        step_result = mpctask.solve_step(step_input);
    catch
        %
        step_result.x_opt = NaN;
        step_result.u_opt = NaN;
        %step_result.sol = NaN;
    end
    %sample_points_X(:,k)
    Data{k,1} = sample_points_X(:,k);
    Data{k,2} = sample_points_Ref(:,k); % constant load over prediction horizon 
    Data{k,3} = step_result.x_opt;
    Data{k,4} = step_result.u_opt;
    % Data{k,5} = J_ocp;
    % Data{k,6} = X_cl;
    % Data{k,7} = U_cl;
    % Data{k,8} = J_cl;
end





end









