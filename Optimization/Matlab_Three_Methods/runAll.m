% 2016-04-06 
% Pattern Recognition(2016-1) 
% Assignment 1: Understanding MLE, MAP, fully Bayesian treatment
% Hong Yeonjung (2015021075)

%% preamble
clear; close all; clc;
%% ##### SET NEW SAMPLE X HERE! ##### %%
x = .3;
%% assign given parameters
N = 100; % sample size
M = 9; % M-th order
X = -1 + 2.*rand(N,1); % random x in the range of [-1 1]
alpha = 5*(10^-3);
beta = 11.1; % precision
sigma = (1/beta)^.5; % standard deviation
noise = normrnd(0, sigma, [N,1]); % normally distributed noise vector
T = sin(2*pi*X) + noise; % target value
%% Maximum Likelihood Estimation (MLE)
% t_MLE: estimated target value by MLE
% opt_w: estimated optimal weight vector
[t_MLE, opt_w] = predictor_MLE(x,X,T,M);


% plot
arbx = -1:.02:1; % arbitrary x vector
arby = zeros(length(arbx),1); % pre-assigning y vector corresponding to arbx
for i = 1:length(arbx)
    arby(i) = curve_fitting(arbx(i), opt_w, M);
end

fig = figure('Position', [100, 100, 1000, 500]);
subplot(1,3,1)
plot(X,T,'o'); hold on % training input data
plot(arbx, sin(2*pi*arbx), 'g-'); hold on  % underlying function of training input, which is y = sin(2*pi*x)
plot(arbx, arby, 'r-', 'linewidth', 3); hold on  % estimated polynomial function resulting from MLE
plot(x, t_MLE, 'k*'); % a point of estimated target value corresponding to the testing x value
text(x, t_MLE, ['\leftarrow x = ' num2str(x) ', t = ' num2str(t_MLE)], 'FontSize', 14);
title('MLE','FontSize', 18); xlabel('x','FontSize', 14); ylabel('t','FontSize', 14); xlim([-1 1]); ylim([-2 2])

%% Maximum A Posteriori (MAP)
% t_MAP: estimated target value by MAP
% opt_w: estimated optimal weight vector
[t_MAP, opt_w] = predictor_MAP(x,X,T,alpha,beta,M);



% plot
arbx = -1:.02:1; % arbitrary x vector
arby = zeros(length(arbx),1); % pre-assigning y vector corresponding to arbx
for i = 1:length(arbx)
    arby(i) = curve_fitting(arbx(i), opt_w, M);
end
subplot(1,3,2)
plot(X,T,'o');hold on % training input data
plot(arbx, sin(2*pi*arbx), 'g-');hold on % underlying function of training input, which is y = sin(2*pi*x)
plot(arbx, arby, 'r-','linewidth', 3);hold on % estimated polynomial function resulting from MAP
plot(x, t_MAP, 'k*') % a point of estimated target value corresponding to the testing x value
text(x, t_MAP, ['\leftarrow x = ' num2str(x) ', t = ' num2str(t_MAP)], 'FontSize', 14);
title('MAP','FontSize', 18); xlabel('x','FontSize', 14); ylabel('t','FontSize', 14); xlim([-1 1]); ylim([-2 2])

%% fully Bayesian
% mean_Bay: the mean of the predictive distribution resulting from a Bayesian treatment
% var_Bay: the variance of the predictive distribution resulting from a Bayesian treatment
[mean_Bay, var_Bay] = predictor_Bayesian(x,X,T,alpha,beta,M);



% plot
arbx = -1:.02:1; % arbitrary x vector
arby = zeros(length(arbx),1); % pre-assigning y vector corresponding to arbx
arbv = zeros(length(arbx),1); % pre-assigning variance vector 
for i = 1: length(arbx)
    [arby(i), arbv(i)] = predictor_Bayesian(arbx(i),X,T,alpha,beta,M);
end
subplot(1,3,3)
plot(X,T,'o'); hold on % training input data
plot(arbx, sin(2*pi*arbx), 'g-'); hold on % underlying function of training input, which is y = sin(2*pi*x)
plot(arbx, arby, 'r-','linewidth', 3); hold on  % mean of the predictive distribution given arbitrary x vector
plot(x, mean_Bay, 'k*') % a point of mean target value corresponding to the testing x value
text(x, mean_Bay, ['\leftarrow x = ' num2str(x) ', t = ' num2str(mean_Bay)],'FontSize', 14)
plot(arbx, arby+arbv.^.5, 'r--'); hold on  % 'mean + std' of the predictive distribution given arbitrary x vector
plot(arbx, arby-arbv.^.5, 'r--'); hold on  % 'mean - std' of the predictive distribution given arbitrary x vector
title('fully Bayesian','FontSize', 18); xlabel('x','FontSize', 14); ylabel('t','FontSize', 14);xlim([-1 1]); ylim([-2 2])
legend('training input data','sin(2\pix)', 'estimated polynomial',  'estimated target value', '\sigma')  % legend