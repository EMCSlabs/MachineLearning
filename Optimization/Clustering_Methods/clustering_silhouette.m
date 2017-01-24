% Applied Math, Dr. Wallraven, Christian, 2015
% clustering basic

clear all
close all
clc

%%
X =[randn(10,4)+ones(10,4); randn(10,4)-ones(10,4)];
figure;
scatter3(X(:,1),X(:,2),X(:,3))
%%
cidx = kmeans(X,2,'distance','sqeuclid');
figure
hold on
scatter3(X(cidx==1,1),X(cidx==1,2),X(cidx==1,3),'rx')
scatter3(X(cidx==2,1),X(cidx==2,2),X(cidx==2,3),'gx')

%%
figure
silhouette(X,cidx,'sqeuclid');
