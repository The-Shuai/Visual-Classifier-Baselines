function C = NNLS( Y, X, maxiter)

% Input
% y           Testing data matrix
% X           Training Data matrix, dim * num
% XTXinv   inv( X'*X+rho/2*eye(N) )
% Par         parameters

% Objective function:
%      min_{A}  ||Y - X * A||_{F}^{2}  s.t.  A>=0

% Notation: 
% y ... (D x 1) the testing data vector where D is the dimension of input
% data
% X ... (D x N) the training data matrix, where D is the dimension of features, and
%           N is the number of training samples.
% a ... (N x 1) is a column vector used to select
%           the most representive and informative samples to represent the
%           input sample y
% Par ...  struture of regularization parameters
[~, M] = size(Y);
[~, N] = size(X);
lambda = 0;
rho = 0.6;
% maxRho = 0.5;
step = 0.5;
minstep = 0.1;


% for rho = [0.1:.1:0.1]
%     Par.rho = rho;
% end
% XTX = inv(X'*X+rho/2*eye(N));
XTXinv = (X'*X+rho/2*eye(N))\eye(N);
XTXinv = (2/rho * eye(N) - (2/rho)^2 * X' / (2/rho * (X * X') + eye(size(X,1))) * X );
% xxx = XTXinv - XTX;
cost = [];


%% initialization
A       = zeros (N, M); % satisfy NN constraint
C       = A;
Delta = C - A;
Par.display = 1;
for iter = 1:maxiter
    
    Cpre = C;
    Apre = A;
    %% update A the coefficient matrix
    A = XTXinv * (X' * Y + rho/2 * C + 0.5 * Delta);
    
    %% update C the data term matrix
    Q = (A - Delta/rho)/(2*lambda/rho+1);
    C = max(0, Q);
    
    %% check the convergence conditions
    stopCA(iter) = max(max(abs(C - A)));
    stopC(iter) = max(max(abs(C - Cpre)));
    stopA(iter) = max(max(abs(A - Apre)));
%     if Par.display %&& (iter==1 || mod(iter,10)==0 || stopC<tol)
%         disp(['iter ' num2str(iter), ...
%             ', max(||c-z||)=' num2str(stopCA(iter),'%2.3e') ...
%             ', max(||c-cpre||)=' num2str(stopC(iter),'%2.3e') ...
%             ', max(||z-zpre||)=' num2str(stopA(iter),'%2.3e')]);
%     end
    cost(iter) = trace(Y'*Y) - 2*trace(Y'*X*A) + trace(A'*X'*X*A);
%     fprintf('Iter = %.2f, cost = %.8f\n',iter,cost(iter));
    
    %% update Deltas the lagrange multiplier matrix
    Delta = (1-step)*Delta + (step) * ( C - A);
    step = max(minstep, step*0.95);
    
    %     %% update rho the penalty parameter scalar
    %     Par.rho = min(1e4, Par.mu * Par.rho);
end

return;