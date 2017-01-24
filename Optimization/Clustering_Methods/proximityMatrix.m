% proximity matrix

%% Load data
load fcmdata.dat
plot(fcmdata(:,1),fcmdata(:,2),'o')

%% calculate proximity 
method = 'euclidean'; % 'chebychev', 'spearman', 'cityblock', 'cosine', ...
prox = pdist(fcmdata,method); % vector
Z = squareform(prox);
imagesc(Z)
