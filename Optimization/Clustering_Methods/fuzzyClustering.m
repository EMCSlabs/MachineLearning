% fuzzy clustering
% fcm - fuzzy c-means clustering

%% Load data
load fcmdata.dat
plot(fcmdata(:,1),fcmdata(:,2),'o')

%% Find 2 clusters using fuzzy c-means clustering.
[centers,U] = fcm(fcmdata,2); 

%% Classify each data point into the cluster with the largest membership value.
maxU = max(U);
index1 = find(U(1,:) == maxU);
index2 = find(U(2,:) == maxU);
hist(U,5)

%% Plot the clustered data and cluster centers.
plot(fcmdata(index1,1),fcmdata(index1,2),'ob')
hold on
plot(fcmdata(index2,1),fcmdata(index2,2),'or')
plot(centers(1,1),centers(1,2),'xb','MarkerSize',15,'LineWidth',3)
plot(centers(2,1),centers(2,2),'xr','MarkerSize',15,'LineWidth',3)
hold off