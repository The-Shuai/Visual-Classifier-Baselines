function [trainLabel,testLabel] = initialization(kernel_train,kernel_test,nclass)
% nclass һ���ж�����
% train_numPerClass ÿһ���ж���ѵ������
% test_numPerClass ÿһ���ж���ѵ������

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


