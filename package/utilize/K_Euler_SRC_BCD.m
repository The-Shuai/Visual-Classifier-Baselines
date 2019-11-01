function [z] = K_Euler_SRC_BCD(C,alpha,A,b)
% BCD �� blockwise coordinate descent �Ż��㷨����д
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% C=(y'X)'=kernel_tstr'=(ts_max_fea'*tr_max_fea)'
% A=tr_max_fea����Ӧ�����ǹ�ʽ�е�X
% b=ts_max_fea����Ӧ�����ǹ�ʽ�е�y
% z����Ӧ�����ǹ�ʽ�е�Sk

row_num = size(C,1); % Sk��Ӧ������,1400
column_num = size(C,2); % Sk��Ӧ������,700
z = ones(row_num,column_num);
m = size(A,1);
N = size(A,2);
beta = 1*1e-4;
obj = zeros(1000,column_num);
dead = zeros(1000);   
for i = 1:column_num
    iter = 1;
%     deadline = 1; % ��ֹ����
%     while deadline  > beta
    while iter  <= 6
        E = z(:,i); % ��iter=2��ʱ��E��ʵ��iter=1ʱ���ֵ
        diag_z = abs(diag(z(:,i)));
        if m > N
            % A\b=inv(A)*b;   b/A=b*inv(A);
            D1 = A' * A * diag_z;
            I1 = eye(size(D1,2));
            % z(:,i) = diag_z * inv(D1 + alpha * I1) * A' * b(:,i);
            z(:,i) = diag_z / (D1 + alpha * I1) * A' * b(:,i);
        else
            D2 = A * diag_z * A';
            I2 = eye(size(D2,2));
            % z(:,i) = diag_z * A' * inv(D2 + alpha * I2) * b(:,i);
            z(:,i) = (diag_z * A') / (D2 + alpha * I2) * b(:,i);
        end
%         obj(iter,i) = (1/2) * (norm(A * z(:,i) - b))^2 + alpha * norm(z(:,i),1);
%         if iter > 1 
%             deadline = norm((z(:,i)-E),1);
%             dead(iter,i) = deadline;
%         end
        iter = iter + 1;
    end
end
end
        



        