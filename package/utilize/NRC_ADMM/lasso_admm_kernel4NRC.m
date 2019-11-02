% Function to perform LASSO regression using Alternating Direction Method
% of Multipliers.
%
% arg min_{B} 0.5*||X - A*B||_{2}^{2} + gamma*||B||_{1}
%
% Usage:- [B,cost] = lasso_admm(X, A, gamma)
%
% where:-
%         b = bias vector
%         lambda = weighting on the l1 penalty
%
%         x = solution
%
% Written by Simon Lucey 2012

function [B,cost] = lasso_admm_kernel4NRC(kernel_tstr, kernel_train, sumY, maxiter)

% Get dimensions of B
c = size(kernel_tstr,2);
r = size(kernel_train,2); 

L = zeros(r,c); % Initialize Lagragian to be nothing (seems to work well)
rho = 0.3; % Set rho to be quite low to start with
maxIter = maxiter; % Set the maximum number of iterations (make really big to ensure convergence)
I = eye(r);
% I = speye(r); % Set the sparse identity matrix
maxRho = 0.1; % Set the maximum mu
C = zeros(r,c); % Initialize C randomly


% Set the fast soft thresholding function
fast_sthresh = @(x,th) sign(x).*max(abs(x) - th,0);

% Set the norm functions
norm2 = @(x) x(:)'*x(:);
norm1 = @(x) sum(abs(x(:))); 
    
cost = [];
fold = 9999999;
fnew = 999999;
n = 0;
% for n = 1:maxIter
% % while (fold-fnew)/fold>1*1e-8 && n<maxIter
while  n<maxIter    
    n = n+1;
    % Solve sub-problem to solve B
    B = (kernel_train+(rho*I/2))\(kernel_tstr + (rho*C/2) + L/2); 
    % Solve sub-problem to solve C
    C = max(0,B - L/rho);
    % Update the Lagrangian
    L = (1-rho)*L + rho*(C - B);  
    %pause; 
    % Section 3.3 in Boyd's book describes strategies for adapting rho
    % main strategy should be to ensure that
%     rho = min(maxRho, rho*1.05); 
    % get the current cost
%     cost(n) = 0.5*norm2(X - A*B) + gamma*norm1(B);
    fold = fnew;
    cost(n) = (sumY+trace(B'*kernel_train*B)-2*trace(kernel_tstr*B'));
    fnew = cost(n);
%     fprintf('Iter = %.2f, cost = %.8f\n',n,cost(n));
end

% while (fold-fnew)/fold>1*1e-8 && n<maxIter    
%     n = n+1;
%     B = (kernel_train+(rho*I/2))\(kernel_tstr + (rho*C/2) - L/2); 
%     C = max(0,B + L/rho);
%     L = (1-rho)*L + rho*(B - C);  
% %     rho = min(maxRho, rho*1.05); 
%     fold = fnew;
%     cost(n) = 0.5*(sumY+trace(B'*kernel_train*B)-2*trace(kernel_tstr*B'));
%     fnew = cost(n);
%     fprintf('Iter = %.2f, cost = %.8f\n',n,cost(n));
% end

