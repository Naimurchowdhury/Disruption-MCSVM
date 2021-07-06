data = X;

[idx,C,sumd] = kmeans(data,4);

figure;
plot(data(:,2),data(:,3),'k*','MarkerSize',.5);
title 'K-means Clustering';
xlabel 'feature 1';
ylabel 'feature 2';