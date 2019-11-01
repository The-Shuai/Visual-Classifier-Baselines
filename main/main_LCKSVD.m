clear;
clc;
options                      =     [];
options. dataset             =     'YaleB_32x32';  %	, PIE_32x32, AR_dataset, usps, MNIST
options.class_num            =     38;
options.tr_num               =     5;
options.val_num              =     10;
options.ts_num               =     0;
options.tr_unlabelled_num    =     0;
options.flag                 =     1;    % 1 represents validation; 2 represents testing
options.seed                 =     1000; % 

addpath('../package/demo/');
addpath('../package/classifier/');
addpath('../package/utilize/');
addpath('../package/utilize/LCKSVD/');
addpath('../package/utilize/LCKSVD/trainingdata/');
addpath('../package/utilize/LCKSVD/OMPbox/');
addpath('../package/utilize/LCKSVD/ksvdbox/');
addpath('../package/graph/');
datasetpath = ['../data/dataset/' options.dataset '.mat'];
load(datasetpath);
accuracy_save = zeros(7,1);

%%LCKSVD
%%% Kernel Parameters
options.kernel.name          =     'linear';
options.kernel.param.gamma   =     2^-2;
options.kernel.param.polyc   =     4;
options.kernel.param.polyd   =     3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options.method.name          =     'LCKSVD';
label_num = options.tr_num * options.class_num; 
options.method.param.iterations = 100; % iteration number
options.method.param.iterations4ini = 20;

for i = 50:20:50
    options.method.param.sparsitythres = i;
    for j = -3:2:1
        options.method.param.sqrt_alpha = 2^(j);
%         options.method.param.sqrt_alpha = j;
        for k = -6:2:-2
            options.method.param.sqrt_beta = 2^(k);
%             options.method.param.sqrt_beta = k;
            for l = 0.7:0.3:0.7
%                 options.method.param.dictsize = l*label_num;
                options.method.param.dictsize = 133;
                iter = 1;
                for ii = 1000:1007
                    options.seed = ii;
                    [~,accuracy(iter)] = demo_classification(fea,gnd,options);
                    iter = iter + 1;
                end 
                fprintf('sparsitythres: %f   sqrt_alpha: %f  sqrt_beta: %f  dictsize: %f   Accuracy: %f\n',options.method.param.sparsitythres,options.method.param.sqrt_alpha,options.method.param.sqrt_beta,options.method.param.dictsize,mean(accuracy));
                fprintf('\n');
            end
        end
    end
end            
