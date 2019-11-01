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

function [B,cost] = lasso_admm_kernel(kernel_tstr, kernel_train, sumY, par)

alpha    = par.method.param.alpha;
rho      = par.method.param.rho;
gd       = par.method.param.gd; 
min_gd   = par.method.param.min_gd;
maxiter  = par.method.maxiter;

% Get dimensions of B
c = size(kernel_tstr,2);
r = size(kernel_train,2); 
L = zeros(r,c); 
I = speye(r); % Set the sparse identity matrix
% max_steplen = 0.5*1e-4; % Set the maximum mu
% maxrho = 3;
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
% while (fold-fnew)/fold>1*1e-8 && n<maxIter
while  n<maxiter
    
    n = n+1;
    % Solve sub-problem to solve B
    B = (kernel_train+rho*I)\(kernel_tstr + rho*C - L); 

    % Solve sub-problem to solve C
    C = fast_sthresh(B + L/rho, alpha/rho); 

    % Update the Lagrangian
%     L = L + steplen*(B - C);  
    L = (1-gd)*L + gd*(B - C);
    %pause; 

    % Section 3.3 in Boyd's book describes strategies for adapting rho
    % main strategy should be to ensure that
    gd = max(min_gd, gd*0.99);
%     steplen = min(max_steplen, steplen*1.05); 

    % get the current cost
%     cost(n) = 0.5*norm2(X - A*B) + gamma*norm1(B);
    fold = fnew;
    cost(n) = 0.5*(sumY+trace(B'*kernel_train*B)-2*trace(kernel_tstr*B'))+ alpha*norm1(B);
    fnew = cost(n);
%     fprintf('Iter = %.2f, cost = %.8f\n',n,cost(n));
end