function [acc_map,accuracy] = demo_classification(fea,gnd,par)

warning off;
acc_map = [];
accuracy = [];
addpath(genpath('package'));
% par.folder_results = ['data/results/' num2str(par.seed) '/' par.dataset];
% if ~isdir(par.folder_results),
%     mkdir(par.folder_results);
% end;
% par.resultmaxpathspm = [par.folder_results '/' par.method.name '_ntrain_' num2str(par.tr_num) '_flag_' num2str(par.flag)];
if strcmp(par.dataset,'MNIST') == 1
    gnd = gnd + 1;
end
if size(fea,2) ~= length(gnd)
    fea = fea';
end

%%%%%%%训练集、测试集、验证集
[tr_fea,tr_label,val_fea,val_label,ts_fea,ts_label,tr_fea_all,tr_fea_unlabelled,tr_all_label]= split_data(fea,gnd,par.seed,par.tr_num,par.val_num,par.ts_num);
tr_fea   =  tr_fea./(repmat(sqrt(sum(tr_fea.*tr_fea)), [size(tr_fea,1),1]));
tr_fea_unlabelled =  tr_fea_unlabelled./(repmat(sqrt(sum(tr_fea_unlabelled.*tr_fea_unlabelled)), [size(tr_fea_unlabelled,1),1]));
tr_fea_all =  tr_fea_all./(repmat(sqrt(sum(tr_fea_all.*tr_fea_all)), [size(tr_fea_all,1),1]));


val_fea  =  val_fea./(repmat(sqrt(sum(val_fea.*val_fea)), [size(val_fea,1),1]));
ts_fea   =  ts_fea./(repmat(sqrt(sum(ts_fea.*ts_fea)), [size(ts_fea,1),1]));
if par.flag == 1,   ts_fea = val_fea; ts_label = val_label;    end

switch par.method.name      
    case 'KCRC'     
        [accuracy,acc_map] = KCRC_classifier(tr_fea,tr_label,ts_fea,ts_label,par);
    case 'KSRC'     
        [accuracy,acc_map] = KSRC_classifier(tr_fea,tr_label,ts_fea,ts_label,par);
    case 'KSRC_ADMM'     
        [accuracy,acc_map] = KSRC_ADMM_classifier(tr_fea,tr_label,ts_fea,ts_label,par);
    case 'K_Euler_SRC'     
        [accuracy,acc_map] = K_Euler_SRC_classifier(tr_fea,tr_label,ts_fea,ts_label,par);
    case 'liblinear'  %1vsrest
        [accuracy,acc_map] = liblinear_classifier(tr_fea,tr_label,ts_fea,ts_label,par);           
    case 'libsvm'    %1vsrest
        [accuracy,acc_map] = libsvm_classifier(tr_fea,tr_label,ts_fea,ts_label,par); 
    case 'KSLRC_L1'     
        [accuracy,acc_map] = KSLRC_L1_classifier(tr_fea,tr_label,ts_fea,ts_label,par); 
    case 'KNRC_ADMM'     
        [accuracy,acc_map] = KNRC_ADMM_classifier(tr_fea,tr_label,ts_fea,ts_label,par); 
    case 'LCKSVD'
        training_feats = tr_fea;
        testing_feats = val_fea;
        [accuracy,acc_map] = LCKSVD_classifier(tr_fea,tr_label,ts_fea,ts_label,training_feats,testing_feats,par);   
    case 'CSDL_KSRC'     
        [accuracy,acc_map] = CSDL_KSRC_classifier(tr_fea,tr_label,ts_fea,ts_label,par);
    case 'LEDL_SRC_ADMM'     
        [accuracy,acc_map] = LEDL_SRC_ADMM_classifier(tr_fea,tr_label,ts_fea,ts_label,par); 
    case 'CDLF_ADMM'     
        [accuracy,acc_map] = HDLN_ADMM_classifier(tr_fea,tr_label,ts_fea,ts_label,par);
    case 'GEDL'     
        [accuracy,acc_map] = GEDL_classifier(tr_fea,tr_label,ts_fea,ts_label,tr_fea_all,tr_fea_unlabelled,par);       
end





