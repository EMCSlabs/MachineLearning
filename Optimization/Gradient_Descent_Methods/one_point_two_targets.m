% gradient descent
% one point two targets
clear; clc; close all
%% initial condition & parameters
initial = [2,2];
lambda1 = 0.1;
lambda2 = 0.1;
maxIter = 30;
x_prev = initial(1);
y_prev = initial(2);

figure(1)
set(gcf,'Position',[1 553 1257 343])
subplot(131)
plot(initial(1),initial(2),'ro')
xlim([initial(1)*-2,initial(1)*2])
ylim([initial(2)*-2,initial(2)*2])
grid minor
[c1,c2] = ginput(2);
hold on; plot(c1,c2,'b*'); hold off

%% cost function & derivatives
% f1 = (c1(1)-initial(1)).^2 + (c2(1)-initial(2)).^2;
% f2 = (c2(1)-initial(1)).^2 + (c2(2)-initial(2)).^2;
syms x y
eval(sprintf('f1 = (%d-x).^2 + (%d-y).^2;',c1(1),c2(1)))
eval(sprintf('f2 = (%d-x).^2 + (%d-y).^2;',c1(2),c2(2)))
df_x1 = diff(f1,x);
df_y1 = diff(f1,y);
df_x2 = diff(f2,x);
df_y2 = diff(f2,y);
f1 = matlabFunction(f1);
f2 = matlabFunction(f2);
df_x1 = matlabFunction(df_x1);
df_y1 = matlabFunction(df_y1);
df_x2 = matlabFunction(df_x2);
df_y2 = matlabFunction(df_y2);


%% main loop
con1 = [];
con2 = [];

for numIter = 1:maxIter
    % update x,y
    x_curr = x_prev - lambda1*df_x1(x_prev) - lambda1*df_x2(x_prev);
    y_curr = y_prev - lambda2*df_y1(y_prev) - lambda2*df_y2(y_prev);
    
    subplot(131)
    plot(x_curr,y_curr,'ro')
    xlim([initial(1)*-2,initial(1)*2])
    ylim([initial(2)*-2,initial(2)*2])
    xlabel(sprintf('numIter: %d',numIter))
    grid minor
    hold on; plot(c1,c2,'b*'); hold off
    
    subplot(132)
    con1 = [con1,f1(x_curr,y_curr)];
    plot(1:numIter,con1,'o-')
    xlim([1 maxIter])
    xlabel(sprintf('error1: %f',f1(x_curr,y_curr)))
    
    subplot(133)
    con2 = [con2,f2(x_curr,y_curr)];
    plot(1:numIter,con2,'o-')
    xlim([1 maxIter])
    xlabel(sprintf('error2: %f',f2(x_curr,y_curr)))
    
    x_prev = x_curr;
    y_prev = y_curr;
    pause(.1)
end



