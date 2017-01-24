function [mean_x, var_x] = predictor_Bayesian(x,X,T,alpha,beta,M)
% This function predicts a mean of estimated target distribution with a single testing x value using Bayesian model.
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
% mean_x: the mean of estimated target distribution resulting from a testing input x
% var_x: the variance of estimated target distribution resulting from a testing input x

% initialize A
A = repmat(X,1, M+1);
for k = 0:M
    A(:, k+1) = A(:, k+1).^k;
end

% phi(x)
phi_x = repmat(x,M+1, 1);
for k = 0:M
    phi_x(k+1) =phi_x(k+1).^k;
end

% S
sigma1 = zeros(size(A,2),size(A,2)); % size(A,2) = M+1
for i = 1:size(A,1)                                % size(A,1) = N = a number of training samples
sigma1 = sigma1 + A(i,:)'*A(i,:);
end
S = inv(alpha*eye(size(A,2)) + beta*sigma1);

% s^2(x)
var_x = 1/beta + (phi_x' * S * phi_x);

% m(x)
sigma2 = zeros(size(A,2), 1); % size(A,2) = M+1
for i = 1:size(A,1)                     % size(A,1) = N = a number of training samples
   sigma2 = sigma2 + A(i,:)'*T(i);
end
mean_x = beta * phi_x' * S * sigma2;

