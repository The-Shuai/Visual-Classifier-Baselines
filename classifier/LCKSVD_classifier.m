function [accuracy2,acc_map] = LCKSVD_classifier(tr_max_fea,tr_label,ts_max_fea,ts_label,training_feats,testing_feats,par)
acc_map = [];
kernel_name      = par.kernel.name;
gamma            = par.kernel.param.gamma;
polyc            = par.kernel.param.polyc;
polyd            = par.kernel.param.polyd;
nclass = length(unique(tr_label));
%%
switch kernel_name
    case 'linear'     
        kernel_train      = tr_max_fea'*tr_max_fea; %相当于公式(15)中的分母，X'X
        kernel_tstr       = ts_max_fea'*tr_max_fea;
        kernel_test       = ts_max_fea'*ts_max_fea;
%         resultmaxpathspm  = [resultmaxpathspm '_' kernel_name '.mat'];
    case 'rbf'
        kernel_train      = sp_dist2((tr_max_fea)',(tr_max_fea)');
        kernel_tstr       = sp_dist2((ts_max_fea)',(tr_max_fea)');
        kernel_test       = sp_dist2((ts_max_fea)',(ts_max_fea)');
        kernel_train      = exp(-gamma*kernel_train); 
        kernel_tstr       = exp(-gamma*kernel_tstr);
        kernel_test       = exp(-gamma*kernel_test);   
%         resultmaxpathspm  = [resultmaxpathspm '_' kernel_name '_' num2str(gamma) '.mat'];
    case 'poly'
        kernel_train      = tr_max_fea'*tr_max_fea;
        kernel_tstr       = ts_max_fea'*tr_max_fea;
        kernel_test       = ts_max_fea'*ts_max_fea;
        kernel_train      = (kernel_train+polyc).^polyd;
        kernel_tstr       = (kernel_tstr+polyc).^polyd;
        kernel_test       = (kernel_test+polyc).^polyd;
%         resultmaxpathspm  = [resultmaxpathspm '_' kernel_name '_c_' num2str(polyc) '_d_' num2str(polyd) '.mat'];     
    case 'Hellinger'
        tr_max_fea        = sign(tr_max_fea).*sqrt(abs(tr_max_fea));
        ts_max_fea        = sign(ts_max_fea).*sqrt(abs(ts_max_fea));
        kernel_train      = tr_max_fea'*tr_max_fea;
        kernel_tstr       = ts_max_fea'*tr_max_fea;
        kernel_test       = ts_max_fea'*ts_max_fea;
%         resultmaxpathspm = [resultmaxpathspm '_' kernel_name '.mat'];  
    case 'chi2'
        tr_max_fea = vl_homkermap(tr_max_fea, 3, 'KCHI2', 'gamma', 1) ;
        ts_max_fea = vl_homkermap(ts_max_fea, 3, 'KCHI2', 'gamma', 1) ;        
        kernel_train      = tr_max_fea'*tr_max_fea;
        kernel_tstr       = ts_max_fea'*tr_max_fea;
        kernel_test       = ts_max_fea'*ts_max_fea; 
%         resultmaxpathspm  = [resultmaxpathspm '_' kernel_name '.mat'];   
    case 'hik'
        tr_max_fea = vl_homkermap(tr_max_fea, 3, 'KINTERS', 'gamma', 1) ;
        ts_max_fea = vl_homkermap(ts_max_fea, 3, 'KINTERS', 'gamma', 1) ;        
        kernel_train      = tr_max_fea'*tr_max_fea;
        kernel_tstr       = ts_max_fea'*tr_max_fea;
        kernel_test       = ts_max_fea'*ts_max_fea; 
%         resultmaxpathspm  = [resultmaxpathspm '_' kernel_name '.mat'];  
    case 'KJS'
        tr_max_fea = vl_homkermap(tr_max_fea, 3, 'KCHI2', 'gamma', 1) ;
        ts_max_fea = vl_homkermap(ts_max_fea, 3, 'KCHI2', 'gamma', 1) ;        
        kernel_train      = tr_max_fea'*tr_max_fea;
        kernel_tstr       = ts_max_fea'*tr_max_fea;
        kernel_test       = ts_max_fea'*ts_max_fea; 
%         resultmaxpathspm  = [resultmaxpathspm '_' kernel_name '.mat'];           
    case 'mlp'
        P1 = 1;
        P2 = -1;
        kernel_train       = tanh(P1*(tr_max_fea'*tr_max_fea)+P2);
        kernel_tstr        = tanh(P1*(ts_max_fea'*tr_max_fea)+P2);
        kernel_test        = tanh(P1*(ts_max_fea'*ts_max_fea)+P2);
%         resultmaxpathspm   = [resultmaxpathspm '_' kernel_name '.mat'];       
    case 'poly_libsvm'
        dot_train         = tr_max_fea'*tr_max_fea;
        dot_tstr          = ts_max_fea'*tr_max_fea;
        dot_test          = ts_max_fea'*ts_max_fea;        
        kernel_train      = dot_train;
        kernel_tstr       = dot_tstr;
        kernel_test       = dot_test;
        for i = 2:polyd
            kernel_train = kernel_train.*(1 + dot_train);
            kernel_tstr  = kernel_tstr.*(1 + dot_tstr);
            kernel_test  = kernel_test.*(1 + dot_test);
        end
%         resultmaxpathspm   = [resultmaxpathspm '_' kernel_name '.mat'];    
end
%%

[H_train,H_test] = initialization(kernel_train,kernel_test,nclass);
addpath(genpath('.\sharingcode-LCKSVD'));
addpath(genpath('.\ksvdbox'));  % add K-SVD box
addpath(genpath('.\OMPbox')); % add sparse coding algorithem OMP

%% constant
sparsitythres = par.method.param.sparsitythres; % sparsity prior
sqrt_alpha = par.method.param.sqrt_alpha; % weights for label constraint term
sqrt_beta = par.method.param.sqrt_beta; % weights for classification err term
dictsize = par.method.param.dictsize; % dictionary size
iterations = par.method.param.iterations; % iteration number
iterations4ini = par.method.param.iterations4ini; % iteration number for initialization

%% dictionary learning process
% get initial dictionary Dinit and Winit
% fprintf('\nLC-KSVD initialization... ');
[Dinit,Tinit,Winit,Q_train] = initialization4LCKSVD(training_feats,H_train,dictsize,iterations4ini,sparsitythres);
% fprintf('done!');

% % run LC K-SVD Training (reconstruction err + class penalty)
% fprintf('\nDictionary learning by LC-KSVD1...');
% [D1,X1,T1,W1] = labelconsistentksvd1(training_feats,Dinit,Q_train,Tinit,H_train,iterations,sparsitythres,sqrt_alpha);
% save('.\dictionarydata1.mat','D1','X1','W1','T1');
% fprintf('done!');

% run LC k-svd training (reconstruction err + class penalty + classifier err)
% fprintf('\nDictionary and classifier learning by LC-KSVD2...')

[D2,X2,T2,W2] = labelconsistentksvd2(training_feats,Dinit,Q_train,Tinit,H_train,Winit,iterations,sparsitythres,sqrt_alpha,sqrt_beta);
save('.\dictionarydata2.mat','D2','X2','W2','T2');
% fprintf('done!');

%% classification process
% [prediction1,accuracy1] = classification(D1, W1, testing_feats, H_test, sparsitythres);
% fprintf('\nFinal recognition rate for LC-KSVD1 is : %.03f ', accuracy1);

[prediction2,accuracy2] = classification(D2, W2, testing_feats, H_test, sparsitythres);
% fprintf('\nFinal recognition rate for LC-KSVD2 is : %.03f ', accuracy2);
end