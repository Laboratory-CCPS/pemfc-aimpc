classdef MyGP


    properties

        Data_all_X % n_data_all x dim_x
        dim_x
        n_data_all

        Data_all_Y % n_data_all x dim_y
        dim_y = 1;


        Data_selected_X % n_data_selected x dim_x
        Data_selected_Y % n_data_selected x dim_y
        %n_data_selected

        Data_not_selected_X % n_data_not_selected x dim_x
        Data_not_selected_Y % n_data_not_selected x dim_y
        %n_data_not_selected

        % note that n_data_all = n_data_selected + n_data_not_selected
        % should hold

        %log_Hyperpara_init
        %log_Hyperpara_opt %only use log_hyp_opt_struct to avoid confusion
        n_hyp_para
        log_hyp_struct
        log_hyp_opt_struct
        data_used_for_hyper_opt
        objective_vals_iter


        kernelfunc
        meanfunc % currently only @meanZero for prediction_fast_ish
        likfunc % currently only @likGauss

        s_noise2
        fixed_noise = false;

        post

        data_transform_info_X = struct('method','none')

        data_transform_info_Y = struct('method','none')
    end % properties

    methods
        function obj = MyGP(Data,pGP,varargin)
            obj.Data_all_X = Data.X;
            obj.Data_not_selected_X = Data.X;
            size_X = size(Data.X);

            obj.dim_x = size_X(2);
            obj.n_data_all=size_X(1);

            obj.Data_all_Y = Data.Y;
            obj.Data_not_selected_Y = Data.Y;



            obj.kernelfunc = pGP.kernelfunc;
            obj.meanfunc = pGP.meanfunc;
            obj.likfunc = pGP.likfunc;


            obj.s_noise2 = pGP.s_noise2;
            
            obj.n_hyp_para = numel(pGP.log_Hyperpara_init);
            obj.log_hyp_struct = struct('mean', [], 'cov', pGP.log_Hyperpara_init, 'lik', log(pGP.s_noise2));
            if nargin == 3
                obj.data_transform_info_X = varargin{1};
            elseif nargin == 4
                obj.data_transform_info_X = varargin{1};
                obj.data_transform_info_Y = varargin{2};
            end

        end % MyGP
        function obj = reset_gp_data(obj)
            obj.Data_not_selected_X = obj.Data_all_X;
            obj.Data_not_selected_Y = obj.Data_all_Y;
            obj.Data_selected_X = [];
            obj.Data_selected_Y = [];
            obj.post = [];

        end % reset_gp_data




        % hyper opti
        function obj = hyperpara_opti(obj,num_trainingdata)

            if num_trainingdata > obj.n_data_all
                error('You do not have that many data')
            end

            [Y_GP_train,idx_train] = datasample(obj.Data_all_Y,num_trainingdata,'Replace',false);%sample randomly form dataset
            X_GP_train = obj.Data_all_X(idx_train,:);

            obj.data_used_for_hyper_opt.Y = Y_GP_train;
            obj.data_used_for_hyper_opt.Y = X_GP_train;

            if obj.fixed_noise
                prior.lik = {{@priorDelta}};
                obj.log_hyp_opt_struct = minimize(obj.log_hyp_struct, @gp, -100, {@infPrior, @infExact, prior}, obj.meanfunc, obj.kernelfunc, obj.likfunc, X_GP_train, Y_GP_train);
            else
                obj.log_hyp_opt_struct = minimize(obj.log_hyp_struct, @gp, -100, @infGaussLik, obj.meanfunc, obj.kernelfunc, obj.likfunc, X_GP_train, Y_GP_train);%
            end % if obj.fixed_noise



        end % hyperpara_opti

        %
        function obj = custom_hyperpara_opti(obj)
           %% Experimental; not fully fleshed out

            %
            %%
            % if you want an initial guess (example) see https://de.mathworks.com/matlabcentral/answers/2093151-how-to-pass-initial-guess-to-ga-this-is-my-sample-code-and-i-want-to-initialize-complex-initial-gu
            %             initial=[1 2];
            % nvars=2;
            % opts=optimoptions('ga','InitialPopulationMatrix',initial,'PopulationSize',1);
            nvars = length(obj.log_hyp_opt_struct.cov);
            A = [];
            b = [];
            Aeq = [];
            beq = [];
            lb = -4.*ones(1,nvars);
            ub = 4.*ones(1,nvars);
            nonlcon = [];
            intcon = [];
            options = optimoptions('ga','PlotFcn', @gaplotbestf,'Display','iter');
            fun = @(decision_vars) custom_objective_function(obj,decision_vars);
            % use ga if you have mixed integer things
            %[decision_vars_opt,fval,exitflag,output,population,scores] = ga(fun,nvars,A,b,Aeq,beq,lb,ub,nonlcon,intcon,options);
           % 
            % options = optimoptions(@fmincon);
            % problem = createOptimProblem('fmincon','objective',...
            % fun,'x0',obj.log_hyp_opt_struct.cov,'lb',lb,'ub',ub,'options',options);
            % ms = MultiStart;
            % [decision_vars_opt,f] = run(ms,problem,20);
            
            %options = optimoptions('particleswarm','SwarmSize',100,'HybridFcn',@fmincon,'UseParallel',true);
            %options = optimoptions('particleswarm','HybridFcn',@fmincon,'UseParallel',true);
            options = optimoptions('particleswarm','Display','iter','UseParallel',true);
            [decision_vars_opt,fval,exitflag,output,points] = particleswarm(fun,nvars,lb,ub,options) 


            obj.log_hyp_opt_struct.cov = decision_vars_opt(1:end);



        end % custom hyperpara opti

        function J = custom_objective_function(obj,decision_vars)
            %% Experimental; not fully fleshed out
            %% Step 1 choose hyperpara/kernefunc etc. (decision variable sof opti problem)
            % reset gp_data from a previous iteration and start again
            obj = obj.reset_gp_data();
            obj.log_hyp_opt_struct.cov = decision_vars(1:end);
           %% step 2 choose points
           % in the gp the num_to_select points are used for online
           % prediction (selecting these points is quite computationally
           % expensive) so for the optoimization maybe do not choose it too
           % large; if you found good hyperparas you can select afterwards
           % again
            num_to_select = 1000;%1000;%200;%50;%200; %
            batch_mode = 1;
            batch_size = 100000;
            obj = choosePoints(obj,num_to_select,batch_mode,batch_size);
           %% step 3 predict GP for all (or some) unselected points
            [y_pred] = predict_fast_ish(obj,obj.Data_not_selected_X);
            y_true = obj.Data_not_selected_Y;
            J = rmse(y_pred,y_true);
                
            


        end %custom_objective_function


        function K = get_kernel_matrix(obj,X,X_tilde)
            K = feval(obj.kernelfunc{:}, obj.log_hyp_opt_struct.cov, X, X_tilde);
        end % get_kernel_matrix

        function Score = get_score(obj,x_true,y_true,p_score)
            % you can get creative here in how you want to score a sample
            % point (performance differences; absolut pred error; distance
            % to selected ....)
            %%


            c_delta_y = p_score.c_delta_y;
            %dist_to_selected = 1;


            %[y_pred,post] = predict_post(obj,x_true); %
            [y_pred,~] = predict_post_mean(obj,x_true); % same as predict_post but avoids predicting things needed for posterior variance

            y_error = abs(y_true-y_pred);
            Score = c_delta_y.*y_error;




        end


        %Predict
        function [y_pred,post] = predict_post(obj,x)

            if isempty(obj.Data_selected_X)
                y_pred = obj.meanfunc(obj.log_hyp_opt_struct,x);
                post = [];
            elseif obj.fixed_noise
                prior.lik = {{@priorDelta}};
                [y_pred, std, ~, ~, ~, post] = gp(obj.log_hyp_opt_struct, {@infPrior, @infExact, prior}, obj.meanfunc, obj.kernelfunc, obj.likfunc, obj.Data_selected_X, obj.Data_selected_Y, x);
            else
                [y_pred, std, ~, ~, ~, post] = gp(obj.log_hyp_opt_struct, @infGaussLik, obj.meanfunc, obj.kernelfunc, obj.likfunc, obj.Data_selected_X, obj.Data_selected_Y, x);
            end % if isempty(obj.Data_selected_X)



        end % predict_post

        function [y_pred,post] = predict_post_mean(obj,x)
            % this function computes the same as predict_post(obj,x),
            % but without predicting the posterior varaince
            if isempty(obj.Data_selected_X)
                y_pred = obj.meanfunc(obj.log_hyp_opt_struct,x);
                post.alpha = [];
            else
                % for equations see GPML book Page 19
                sigma_n = exp(obj.log_hyp_opt_struct.lik);
                sz_x = size(obj.Data_selected_X);
                num_selected = sz_x(1);
                K = get_kernel_matrix(obj,obj.Data_selected_X,obj.Data_selected_X);
                L = chol(K+sigma_n^2*eye(num_selected),'lower');
                post.alpha = L'\(L\obj.Data_selected_Y);
                K_pred = get_kernel_matrix(obj,x,obj.Data_selected_X);
                y_pred = K_pred*post.alpha;

            end % if isempty(obj.Data_selected_X)

        end % predict_post_mean(obj,x)


        function obj = choosePoints(obj,num_to_select,varargin)
            %Note/Idea:
            % we want to include the data points in the gp which maximize
            % a defined score/objective/"aquisition" function
 

            %if false we always choose among all data; if true we choose among a randomly sampled set (resampled each time we choose)
            if nargin == 2
                batch_mode = 0;
            else
                batch_mode = varargin{1};
            end

            if batch_mode % batch mode is experimental and not fully fleshed out
                batchsize = varargin{2}; 
            end
            



            p_score.c_delta_y = 1;




            for k = 1:num_to_select
               k
                % predict GP for all not yet selected point
                %y_pred_all = obj.predict_post(obj.Data_not_selected_X);
                if batch_mode
                    % keep track of inidzes
                    [Data_not_selected_Y_batch,idx_not_selected_global] = datasample(obj.Data_not_selected_Y,batchsize,'Replace',false);
                    Data_not_selected_X_batch = obj.Data_not_selected_X(idx_not_selected_global,:);
                else
                    % keep track of indices
                    idx_not_selected_global = 1:length(obj.Data_not_selected_Y);
                    Data_not_selected_X_batch = obj.Data_not_selected_X; %
                    Data_not_selected_Y_batch = obj.Data_not_selected_Y;

                end


                Score = get_score(obj,Data_not_selected_X_batch,Data_not_selected_Y_batch,p_score);

                % select point which maximizes the Score
                [~,idx_max_Score] = max(Score);
                obj.Data_selected_X(k,:) = Data_not_selected_X_batch(idx_max_Score,:);
                obj.Data_selected_Y(k,1) = Data_not_selected_Y_batch(idx_max_Score,:);

                % remove selected from not selected (from whole set)

                obj.Data_not_selected_X(idx_not_selected_global(idx_max_Score),:) = [];
                obj.Data_not_selected_Y(idx_not_selected_global(idx_max_Score),:) = [];

            end % for k = 1:num_to_select
            % select random points on top of scoring based points:
            if nargin == 5
                
                additionally_select_random = varargin{3};
                if additionally_select_random > 0
                    % additionally_select_random points:
                    [Data_not_selected_Y_batch,idx_not_selected_global] = datasample(obj.Data_not_selected_Y,additionally_select_random,'Replace',false);
                    Data_not_selected_X_batch = obj.Data_not_selected_X(idx_not_selected_global,:);
                    % add to selected points:
                    obj.Data_selected_X = [obj.Data_selected_X; Data_not_selected_X_batch];
                    obj.Data_selected_Y = [obj.Data_selected_Y; Data_not_selected_Y_batch];
                    %remove selected from not selected:
                    obj.Data_not_selected_X(idx_not_selected_global,:) = [];
                    obj.Data_not_selected_Y(idx_not_selected_global,:) = [];
    
                end

            end% if nargin == 4  additionally select random points


            % compute post struc
            [~,obj.post] = predict_post_mean(obj,zeros(1,obj.dim_x));

        end % choosePoints



        function [y_pred] = predict_fast_ish(obj,x)
            % this needs an obj.post.alpha to be defined

            % - currently assumes zero mean prior!
            K_pred = get_kernel_matrix(obj,x,obj.Data_selected_X);
            y_pred = K_pred*obj.post.alpha;

            %%



        end % predict_fast_ish

        % plot 2D
        function plotGP_2D(obj,gridsize)
            %% plots results in detransformed data "coordinates"
  

            % detransform to original "coordinates"
            Data_all_X_detrans = detransform_data(obj.Data_all_X,obj.data_transform_info_X);
            Data_selected_X_detrans = detransform_data(obj.Data_selected_X,obj.data_transform_info_X);

            Data_all_Y_detrans = detransform_data(obj.Data_all_Y,obj.data_transform_info_Y);
            Data_selecetd_Y_detrans = detransform_data(obj.Data_selected_Y,obj.data_transform_info_Y);

            figure()
            hold on
            plot3(Data_all_X_detrans(:,1),Data_all_X_detrans(:,2),Data_all_Y_detrans,'kx')
            plot3(Data_selected_X_detrans(:,1),Data_selected_X_detrans(:,2),Data_selecetd_Y_detrans,'ro','MarkerSize',10,'MarkerFaceColor','red')

            x1_range_detrans = linspace(min(Data_all_X_detrans(:,1)),max(Data_all_X_detrans(:,1)),gridsize);
            x2_range_detrans = linspace(min(Data_all_X_detrans(:,2)),max(Data_all_X_detrans(:,2)),gridsize);
            x1_range = linspace(min(obj.Data_all_X(:,1)),max(obj.Data_all_X(:,1)),gridsize);
            x2_range = linspace(min(obj.Data_all_X(:,2)),max(obj.Data_all_X(:,2)),gridsize);

            [X1,X2] = meshgrid(x1_range,x2_range);
            [X1_detrans,X2_detrans] = meshgrid(x1_range_detrans,x2_range_detrans);
            for i = 1:length(X1)
                for j = 1:length(X1)

                    Y_GP_detrans(i,j) =  detransform_data(predict_fast_ish(obj,[X1(i,j) X2(i,j)]),obj.data_transform_info_Y);
                end % j
            end % i
            surf(X1_detrans,X2_detrans,Y_GP_detrans,'FaceAlpha',0.5 )



        end% plotGP_2D


        %plot 1D
        function plotGP_1D(obj,gridsize)
            %% plots results in detransformed data "coordinates"
            figure()
            hold on
            Data_all_X_detrans = detransform_data(obj.Data_all_X,obj.data_transform_info_X);
            Data_selected_X_detrans = detransform_data(obj.Data_selected_X,obj.data_transform_info_X);

            Data_all_Y_detrans = detransform_data(obj.Data_all_Y,obj.data_transform_info_Y);
            Data_selected_Y_detrans = detransform_data(obj.Data_selected_Y,obj.data_transform_info_Y);

            plot(Data_all_X_detrans(:,1),Data_all_Y_detrans,'kx')
            plot(Data_selected_X_detrans(:,1),Data_selected_Y_detrans,'ro','MarkerSize',10,'MarkerFaceColor','red')



            x1_range = linspace(min(obj.Data_all_X(:,1)),max(obj.Data_all_X(:,1)),gridsize);
            x1_range_detrans = linspace(min(Data_all_X_detrans(:,1)),max(Data_all_X_detrans(:,1)),gridsize);
            for k = 1:length(x1_range)
                Y_GP_detrans(k) =  detransform_data(predict_fast_ish(obj,[x1_range(k)]),obj.data_transform_info_Y);
            end

            plot(x1_range_detrans,Y_GP_detrans,'r--','LineWidth',1)

            xlabel('x')
            ylabel('y')
            legend('all data','used data','GP post mean')

        end % plotGP_1D

        %saveobj
        function s_obj = saveobj(obj)
            s_obj = obj;
        end





    end % methods
    methods (Static)
        %loadobj
        function obj = loadobj(s_obj)
            obj = s_obj;
        end %loadobj



    end % methods (Static)
end % classdef