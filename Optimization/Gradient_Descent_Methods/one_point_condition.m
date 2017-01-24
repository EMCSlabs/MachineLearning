% gradient descent
% one point condition
clear;close all;clc
%% initial condition & parameters
initial = [2,2];
lambda1 = 0.1;
lambda2 = 0.1;
maxIter = 30;
x_prev = initial(1);
y_prev = initial(2);

figure
plot(initial(1),initial(2),'ro')
xlim([initial(1)*(-2) initial(1)*2])
ylim([initial(2)*(-2) initial(2)*2])
grid minor
[c1,c2] = ginput(1);
hold on; plot(c1,c2,'b*'); hold off

%% cost function & derivatives
% f = (target(1)-initial(1)).^2 + (target(2)-initial(2)).^2;
syms x y
eval(sprintf('f = (%d - x).^2 + (%d - y).^2;',c1,c2))
df_x = diff(f,x);
df_y = diff(f,y);
f = matlabFunction(f);
df_x = matlabFunction(df_x);
df_y = matlabFunction(df_y);

%% main loop
con = [];
for numIter = 1:maxIter
    % update x,y coordinates
    x_curr = x_prev - lambda1*df_x(x_prev);
    y_curr = y_prev - lambda1*df_y(y_prev);
    
    % plot
    plot(x_curr,y_curr,'ro'); hold on
    xlim([initial(1)*(-2) initial(1)*2])
    ylim([initial(2)*(-2) initial(2)*2])
    plot(c1,c2,'b*'); hold off
    xlabel(sprintf('numIter: %d',numIter))
    grid minor
    
    x_prev = x_curr;
    y_prev = y_curr;
    
    pause(.1)
end
