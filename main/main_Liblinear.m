clear;
clc;
options                      =     [];
options. dataset             =     'YaleB_Gaussian_6';  %	, PIE_32x32, AR_dataset, usps, MNIST
options.tr_num               =     5;
options.val_num              =     10;
options.ts_num               =     0;
options.tr_unlabelled_num    =     0;
options.flag                 =     1;            % 1 represents validation; 2 represents testing
options.seed                 =     1000;

options.method.name          =     'liblinear';


addpath('../package/demo/');
addpath('../package/classifier/');
addpath('../package/utilize/');
addpath('../package/utilize/liblinear-2.20/');
addpath('../package/utilize/liblinear-2.20/matlab/');

datasetpath = ['../data/dataset/' options.dataset '.mat'];
load(datasetpath);
fea = double(fea);
accuracy = zeros(1,8);
%%%%%%%其他方法参数、svm
for i =-4:1:15
    options.method.param.c       =     2^(i);
    iter = 1;
    for ii = 1000:1007
        options.seed = ii;
        [~,accuracy(iter)] = demo_classification(fea,gnd,options);
%     predict_accuracy(:,:,iter) = demo_classification(fea,gnd,options);
        iter = iter + 1;
    end 
    fprintf('c: %f    Accuracy: %f\n',options.method.param.c, mean(accuracy));
end
fprintf('\n');

