function [H] = KSRC_BCD(C,B,alpha,sumY,maxiter)
nBases = size(C,1);
H = zeros(nBases,size(C,2));
B_revise = tril(B,-1)+triu(B,1);

iter = 0;
fold = 9999999;
fnew = 999999;
while (fold-fnew)/fold>1*1e-6 && iter<maxiter
    iter = iter+1;
    %update H 
    for i=1:nBases
        H_tp = C(i,:)-B_revise(i,:)*H;
        H(i,:) = (max(H_tp,alpha)+min(H_tp,-alpha))/B(i,i);
    end 
    fold = fnew;
    fnew1 = sumY+trace(H'*B*H)-2*trace(C*H');
    fnew2 = 2*alpha*(sum(sum(abs(H))));
    fnew = fnew1+fnew2;
%     if mod(iter, 100) == 0
%         fprintf('iter = %d rec_err = %.5f\n',iter,fnew);  
%     end
end
return;