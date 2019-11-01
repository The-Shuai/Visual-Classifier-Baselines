clear;
clc;
options                      =     [];
options. dataset             =     'USPS';  %	, PIE_32x32, AR_dataset, usps, MNIST
options.tr_num               =     5;
options.val_num              =     100;
options.ts_num               =     0;
options.tr_unlabelled_num    =     0;
options.flag                 =     1;            % 1 represents validation; 2 represents testing
options.seed                 =     1000;

addpath('../package/demo/');
addpath('../package/classifier/');
addpath('../package/utilize/');
addpath('../package/graph/');
datasetpath = ['../data/dataset/' options.dataset '.mat'];
load(datasetpath);
fea = double(fea);
accuracy = zeros(1,8);


%%SRC_ADMM
%%% Kernel Parameters
options.kernel.name          =     'linear';
options.kernel.param.gamma   =     2^-2;
options.kernel.param.polyc   =     4;
options.kernel.param.polyd   =     3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options.method.name            =     'KSRC_ADMM';
options.method.param.min_gd    =     0.0001;
options.method.maxiter         =     300;
for i = -4:1:-4
    options.method.param.alpha = 2^(i);   
    for j = 1:0.2:1
        options.method.param.rho = j;
        % 为了保证收敛，要让rho>=gd
        for k = 0.5:0.5:0.5
            if k > options.method.param.rho
                break;
            else
                options.method.param.gd = k;
            end
            iter = 1;
            for ii = 1000:1007
                options.seed = ii;
                [~,accuracy(iter)] = demo_classification(fea,gnd,options);
                iter = iter + 1;
            end 
            fprintf('alpha: %f   rho: %f  gd: %f Accuracy: %f\n',options.method.param.alpha,options.method.param.rho,options.method.param.gd,mean(accuracy));
%             fprintf('\n');
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
