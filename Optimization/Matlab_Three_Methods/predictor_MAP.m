function [t, opt_w] = predictor_MAP(x,X,T,alpha,beta,M)
% This function predicts a single target value with a single testing x value using MAP.
%
% INPUT
% x: testing input
% X: training input vector
% T: training target vector
% alpha: precision of normal distribution of weight
% beta: precision of normal distribution of T
% M: order of polynomial function
%
% OUTPUT
% t: estimated target value corresponding to 'x'
% opt_w: optimal weight vector minimizing a loss function


% initialize A
A = repmat(X,1, M+1);
for k = 0:M
    A(:, k+1) = A(:, k+1).^k;
end

% Sum of squared errors + regularization term, which is loss function
MSE =@(w) .5*beta*sum((A*[w(1); w(2); w(3); w(4); w(5); w(6); w(7); w(8); w(9); w(10)] - T).^2) +...
    .5*alpha*[w(1); w(2); w(3); w(4); w(5); w(6); w(7); w(8); w(9); w(10)]'*[w(1); w(2); w(3); w(4); w(5); w(6); w(7); w(8); w(9); w(10)];

% set optimiziatioin solver, algorithm, maximum iterations
options = optimoptions(@fminunc,'Algorithm','quasi-newton','MaxIter',1500);
% random starting point
init = rand(10,1);
% optimal weight vector
opt_w = fminunc(MSE, init, options);

% phi of a testing x 
phi = repmat(x,1, M+1);

t = 0;
for k = 0:M
    t = t + opt_w(k+1) * (phi(k+1).^k); % sum of (weight and phi of testing x)
end