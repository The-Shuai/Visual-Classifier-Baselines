function [trainLabel,testLabel] = initialization(kernel_train,kernel_test,nclass)
% nclass 一共有多少类
% train_numPerClass 每一类有多少训练样本
% test_numPerClass 每一类有多少训练样本

train_numPerClass = round(size(kernel_train,2)/nclass);
test_numPerClass = round(size(kernel_test,2)/nclass);

testLabel = [];
trainLabel = [];

for class = 1:nclass
    labelvector = zeros(nclass,1);
    labelvector(class) = 1;
    testLabel = [testLabel repmat(labelvector,1,test_numPerClass)];
    trainLabel = [trainLabel repmat(labelvector,1,train_numPerClass)];
end


