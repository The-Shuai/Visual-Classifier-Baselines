clear;
clc;
options                      =     [];
options. dataset             =     'UCMerced_imagenet-resnet-50-dag';  
options.class_num            =     5;
options.tr_num               =     10;
options.val_num              =     5;
options.ts_num               =     0;
options.tr_unlabelled_num    =     0;
options.flag                 =     1;            % 1 represents validation; 2 represents testing
options.seed                 =     1000;

addpath('../package/demo/');
addpath('../package/classifier/');
addpath('../package/utilize/');
addpath('../package/utilize/LEDL/');
addpath('../package/graph/');
datasetpath = ['../data/dataset/' options.dataset '.mat'];
load(datasetpath);
fea = double(fea);
accuracy = zeros(1,8);





%%LEDL_SRC_ADMM
%%% Kernel Parameters
options.kernel.name          =     'linear';
options.kernel.param.gamma   =     2^-2;
options.kernel.param.polyc   =     4;
options.kernel.param.polyd   =     3;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options.method.name          =     'LEDL_SRC_ADMM';
label_num = options.tr_num * options.class_num;
options.method.param.min_gd_train    =     0.0001; % 最小步长
options.method.param.min_gd_test     =     0.0001; % 迭代次数
options.method.maxiter               =     10;
% options.method.param.rho_train = 1; % 为了保证收敛，要让rho>=gd
% options.method.param.gd_train = 0.5;
% options.method.param.rho_test = 1;
% options.method.param.gd_test = 0.5;

for i = 1:-0.5:1  % 设置字典大小
    options.method.param.K = label_num * i;
%     options.method.param.K = 133;
    % 设置训练字典时的参数
    for j = -12:3:-3 
        options.method.param.alpha_train = 2^(j);
        options.method.param.alpha_train = 0.000001;
        for k = -12:3:-3
            options.method.param.lambda = 2^(k);
            options.method.param.lambda = 0.2;
            for l = -14:2:-14
%                 options.method.param.omega = 2^(l);
                options.method.param.omega = 0;
                for m = 1:0.5:1 % 为了保证收敛，要让rho>=gd
                    options.method.param.rho_train = m;
                    for n =0.5:0.5:0.5
                        if n > options.method.param.rho_train
                            break;
                        else
                            options.method.param.gd_train = n;
                        end
                        % 设置测试时的参数
                        for o = -12:3:-3 
                            options.method.param.alpha_test = 2^(o);
                            options.method.param.alpha_test = 50;
                            for p = 1:0.5:1
                                options.method.param.rho_test = p;
                                for q =0.5:0.5:0.5
                                    if q > options.method.param.rho_test
                                        break;
                                    else
                                        options.method.param.gd_test = q;
                                    end
                                    iter = 1;
                                    for ii = 1000:1000
                                        options.seed = ii;
                                        [~,accuracy(iter)] = demo_classification(fea,gnd,options);
                                    %     predict_accuracy(:,:,iter) = demo_classification(fea,gnd,options);
                                        iter = iter + 1;
                                    end 
                                    fprintf('alpha_train: %f   lambda: %f   omega: %f    %f\n',options.method.param.alpha_train,options.method.param.lambda,options.method.param.omega);  
                                    fprintf('alpha_test: %f    K:  %f   mean_Accuracy: %f\n',options.method.param.alpha_test,options.method.param.K,mean(accuracy)); 
                                    fprintf('\n');
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%