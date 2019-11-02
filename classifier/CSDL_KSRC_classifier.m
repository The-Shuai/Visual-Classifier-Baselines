function [accuracy,acc_map] = CSDL_KSRC_classifier(tr_max_fea,tr_label,ts_max_fea,ts_label,par)

alpha            = par.method.param.alpha;
tau              = par.method.param.tau;
maxiter          = par.method.maxiter;
nBases           = par.method.nBases;
kernel_name      = par.kernel.name;

nclass = length(unique(tr_label));
% fprintf('compute kernel for train&test\n');
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
%class specific dictionary learning
%training stage
% fprintf('Training stage: Dictionary Learning\n'); 

W = CSDL_KSRC_DL(kernel_train,tr_label,nBases,alpha/(1+tau),maxiter);

% fprintf('Testing stage: Sparse Representation\n');   
tr_blocks = cell(1, nclass);
for class = 1: nclass
    tr_blocks{class} = kernel_train(tr_label == class, tr_label == class);
end
tr_block_diag_mat = blkdiag(tr_blocks{:});
B = (W'*(tau*tr_block_diag_mat+kernel_train))*W;
Codes_s = KSRC_BCD(W'*((1+tau)*kernel_tstr)',B,alpha,(1+tau*nclass)*trace(kernel_test),maxiter);
% B = (W'*kernel_train)*W;   
% Codes_s = KSRC_BCD(W'*kernel_tstr',B,alpha,trace(kernel_test),maxiter);
rec_err = zeros(nclass,length(ts_label));  
Codes = W*Codes_s; 

for class = 1:nclass
    s = Codes(tr_label == class,:);
    ker_tstr = kernel_tstr(:,tr_label == class);
    ker_train = kernel_train(tr_label == class, tr_label == class);
    rec_err(class,:) = diag(kernel_test) - 2*diag(ker_tstr*s) + diag((s'*ker_train)*s);
end
[~,ID] = min(rec_err);
ID = ID';
acc_map = zeros(nclass,1);
acc_acc = zeros(nclass,1);
for jj = 1 : nclass
    idx = find(ts_label == jj);
    curr_pred_label = ID(idx);
    curr_gnd_label = ts_label(idx);    
    acc_map(jj) = (length(find(curr_pred_label == curr_gnd_label)))/(length(idx));
    acc_acc(jj) = (length(find(curr_pred_label == curr_gnd_label)))/(max(length(find(ID == jj)),1e-8));
end  
avr_map = mean(acc_map);      
% fprintf('MeanAP: %f\n',avr_map);
accuracy = mean(acc_acc);      
% fprintf('Accuracy: %f\n',avr_acc); 
% save(resultmaxpathspm,'acc_map','acc_acc','avr_map','avr_acc');