clear;
clc;
options                      =     [];
options. dataset             =     'YaleB_Gaussian_2';  %	, PIE_32x32, AR_dataset, usps, MNIST
options.tr_num               =     5;
options.val_num              =     10;
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

%%CRC
%%% Kernel Parameters
options.kernel.name          =     'linear';
options.kernel.param.gamma   =     2^-2;
options.kernel.param.polyc   =     4;
options.kernel.param.polyd   =     3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options.method.name            =     'KCRC';
for i = -12:1:1
    options.method.param.beta = 2^(i);   
    iter = 1;
    for ii = 1000:1007
        options.seed = ii;
        [~,accuracy(iter)] = demo_classification(fea,gnd,options); 
        iter = iter + 1;
    end
    fprintf('beta: %f   Accuracy: %f\n',options.method.param.beta,mean(accuracy));
end  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%