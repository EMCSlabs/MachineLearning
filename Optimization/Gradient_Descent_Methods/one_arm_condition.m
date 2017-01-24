% gradient descent
% one arm condition

clear;clc;close all

%% initial condition & parameters
xTarget = 5;    % target x coordinate
yTarget = 3;    % target y coordinate
lambda = .0001;          % learning rate
theta_prev = 45*pi/180;  % initial theta
maxIter = 1000;        % maximum iteration

%% define cost function & derivative
syms x
eval(sprintf('f = (5*sin(x)-%d).^2 + (5*cos(x)-%d).^2;',xTarget,yTarget))
df = diff(f,x);
fh = matlabFunction(df); % derivative
f = matlabFunction(f); % cost function

%% main loop
con = [];
for numIter = 1:maxIter
    % update theta
    theta_curr = theta_prev - lambda*fh(theta_prev); 
    
    % plot error
    con = [con,f(theta_curr)];
    plot(1:numIter,con,'-o');
    xlim([1 maxIter])
    ylim([0 max(con)])
    xlabel(sprintf('Iteration: %d, Error: %.4f',numIter,f(theta_curr)))
    
    theta_prev = theta_curr;
    pause(.01)
end

