% Applied Math, Dr. Wallraven, Christian, 2015
% clustering basic

clear all
close all
% Data
% how many points
n = 60;
% how large the radius of the circle should be
radius = rand;
xc = 0;
yc = 0;
% generate random points in a circle
theta = rand(1,n)*(2*pi);
r = sqrt(rand(1,n))*radius;
x1 = xc + r.*cos(theta);
y1 = yc + r.*sin(theta);

% Increase the size of the circle
radiusFactor=2;
radius = radiusFactor*radius;
% Keep the density of points the same
n = n*radiusFactor*radiusFactor;

% offset for second circle
offset=2*radius;

% generate points in the second circle
theta = rand(1,n)*(2*pi);
r = sqrt(rand(1,n))*radius;
x2 = xc + r.*cos(theta)+offset;
y2 = yc + r.*sin(theta);

% Check the two circles
plot(x1,y1,'b.');
hold on
plot(x2,y2,'r.');
axis equal

% create the data for the two circles
data=[x1' y1'];
data=[data; [x2' y2']];

save('circles_clean.mat','data');

%% visualize distance matrix (default Euclidean)
pf=pdist(data);  % pairwise distance
figure;
imagesc(squareform(pf));

% perform three different clusterings
Zs=linkage(data,'single','euclidean');
Zc=linkage(data,'complete','euclidean');
Za=linkage(data,'average','euclidean');

% dendrogram
figure;
subplot(131); dendrogram(Zs)
subplot(132); dendrogram(Zc)
subplot(133); dendrogram(Za)

% make two clusters for each clustering method
T={};

T{1}=cluster(Zs,'maxclust',2);
T{2}=cluster(Zc,'maxclust',2);
T{3}=cluster(Za,'maxclust',2);

titles={'Single Linkage (MIN)','Complete Linkage (MAX)','Group Average'};

% plot all three clustering solutions
for i=1:3
    figure;
    subplot(1,2,1);
    plot(data(:,1),data(:,2),'k.');
    axis equal
    subplot(1,2,2);
    
    inds1=T{i}==1;
    inds2=T{i}==2;
    
    plot(data(inds1,1),data(inds1,2),'b.');
    hold on
    plot(data(inds2,1),data(inds2,2),'r.');
    axis equal
    title(titles{i})
end

%% do clustering on noisy data!

% shift some random indices of the big data circle
indices=randi(length(x2),20,1);
data=[x1' y1'];
data=[data; [x2' y2']];
data=[data; [x2(indices)'-radius y2(indices)']];

save('circles_noisy.mat','data');

% plot solutions for two clusters on noisy data
figure;
plot(data(:,1),data(:,2),'k.');

Zs=linkage(data,'single','euclidean');
Zc=linkage(data,'complete','euclidean');
Za=linkage(data,'average','euclidean');

T={};

T{1}=cluster(Zs,'maxclust',2);
T{2}=cluster(Zc,'maxclust',2);
T{3}=cluster(Za,'maxclust',2);


for i=1:3
    figure;
    subplot(1,2,1);
    plot(data(:,1),data(:,2),'k.');
    axis equal
    subplot(1,2,2);
    inds1=T{i}==1;
    inds2=T{i}==2;
    
    plot(data(inds1,1),data(inds1,2),'b.');
    hold on
    plot(data(inds2,1),data(inds2,2),'r.');
    axis equal
    title(titles{i})
end


