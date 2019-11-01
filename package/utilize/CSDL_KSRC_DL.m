function [W] = CSDL_KSRC_DL(kernel_train,tr_label,nBases,alpha,maxiter)
nclass = length(unique(tr_label));
N      = size(kernel_train(tr_label == nclass, tr_label == nclass),2);
W      = zeros(size(kernel_train,2),nclass*nBases);
wc     = zeros(N,nBases,nclass);
rand('seed',1);
wt = rand(N,nBases) - 0.5;
for class = 1:nclass
    for j = 1:nBases
        lambda = sqrt(wt(:,j)'*kernel_train(tr_label == class, tr_label == class)*wt(:,j));
        wc(:,j,class) = wt(:,j)/(lambda);
    end  
end
hc = zeros(nBases,N,nclass);
EPS = 1e-16;
fold = 9999999;
fnew = 999999;
iter = 0;
while abs((fold-fnew)/fold)>1*1e-16 && iter<maxiter
tic
t0 = cputime;
% while iter<1000

    iter = iter+1;
    
    %update hc
    for class = 1:nclass
        C = wc(:,:,class)'*kernel_train(tr_label == class, tr_label == class);
        B = C*wc(:,:,class); 
        B_revise = tril(B,-1)+triu(B,1);           
        for i=1:nBases
            h_t = C(i,:)-B_revise(i,:)*hc(:,:,class);
            hc(i,:,class) = ((max(h_t,alpha)+min(h_t,-alpha)))/B(i,i);
        end        
    end
    
    %update the objective function
    fold = fnew;  
    fnew = 0;    
    for class = 1:nclass
        C = wc(:,:,class)'*kernel_train(tr_label == class, tr_label == class);
        fnew1 = trace(kernel_train(tr_label == class, tr_label == class))-2*trace(C*hc(:,:,class)')+trace((hc(:,:,class)*hc(:,:,class)')*(C*wc(:,:,class)));
        fnew2 = 2*alpha*(sum(sum(abs(hc(:,:,class)))));
        fnew = fnew + fnew1+fnew2;
    end
%     if mod(iter, 500) == 0    
%         fprintf('Iteration = %.2f  ',iter);%display which iteration    
%         fprintf('relative error = %.10f \n',fnew);
%     end    
    
    %update wc
    hc(hc==0) = EPS; 
    for class = 1:nclass
        F = hc(:,:,class)*hc(:,:,class)';
        F_revise = tril(F,-1)+triu(F,1);
        for j = 1:nBases
            wc(:,j,class) = (hc(j,:,class)'-wc(:,:,class)*F_revise(:,j));
            lambda = sqrt(wc(:,j,class)'*kernel_train(tr_label == class, tr_label == class)*wc(:,j,class));
            wc(:,j,class) = wc(:,j,class)/lambda;
        end   
    end
    hc = (abs(hc)>2*EPS).*hc;  
end
% t1 = cputime-t0;
% aaaaa = num2str(toc);

nCount = 0;
for class = 1:nclass
    W((nCount+1):(nCount+length(find(tr_label == class))),((class-1)*nBases+1):(class*nBases)) = wc(:,:,class);
    nCount = nCount + length(find(tr_label == class));
end
% t2 = cputime-t0;
% bbbbb = num2str(toc);
end
