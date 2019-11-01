function [tr_fea,tr_label,val_fea,val_label,ts_fea,ts_label,tr_fea_all,tr_fea_unlabelled,tr_all_label] = split_data(fea,gnd,seed,tr_num,val_num,ts_num)
% tr_idx 训练集有标签数据的标号
% tr_fea 训练集有标签的数据
% idx_unlabelled 训练集无标签数据的标号
% tr_fea_unlabelled 训练集无标签的数据
% val_idx 验证集标号
% val_fea 验证集数据
% ts_idx 测试集标号
% ts_fea 测试集数据
% idx_labelled 整个数据集有标签数据的标号，由三部分组成，idx_labelled = [tr_idx',val_idx',ts_idx']'
% idx_all 整个数据集所有数据的标号，由两部分组成，idx_all = [tr_idx',idx_unlabelled']'
% tr_fea_all 整个训练集所有数据

tr_idx  = [];
val_idx = [];
ts_idx  = [];
clabel = unique(gnd);
for jj = 1:length(clabel)
    idx_label = find(gnd == jj);
%     idx_unlabelled = idx_label;
    num = single(length(idx_label));
    rand('seed',double(jj)+(seed-1)*length(clabel));
    idx_rand = randperm(num);
    tr_idx  = [tr_idx;(idx_label(idx_rand(1:tr_num)))];
    val_idx = [val_idx;(idx_label(idx_rand((tr_num+1):(tr_num+val_num))))];
%     tr_unlabelled_idx = [tr_unlabelled_idx;(idx_label(idx_rand((val_num+tr_num+1):(tr_unlabelled_num+tr_num+val_num))))];
    if ts_num < (num-(tr_num+val_num+ts_num))
        ts_idx = [ts_idx; (idx_label(idx_rand((tr_num+val_num+1):(tr_num+val_num+ts_num))))];
    else
        ts_idx = [ts_idx; (idx_label(idx_rand((tr_num+val_num+1):end)))];
    end    
end

idx_labelled = [tr_idx',val_idx',ts_idx']';
idx_unlabelled = find(gnd);
idx_unlabelled(idx_labelled,:) = [];
idx_all = [tr_idx',idx_unlabelled']';

tr_fea = fea(:,tr_idx);
val_fea = fea(:,val_idx);
ts_fea = fea(:,ts_idx);
tr_fea_unlabelled = fea(:,idx_unlabelled);
tr_fea_all = fea(:,idx_all);

tr_label = gnd(tr_idx);
val_label = gnd(val_idx);
ts_label = gnd(ts_idx);
tr_all_label = gnd(idx_all);

