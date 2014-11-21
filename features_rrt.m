

%% DESCRIPTION

clc;
clear all;

% Transforms features into Recht and Rahimi’s Random Fourier Feature as defined in:
% Rahimi, Ali, and Benjamin Recht. "Random features for large-scale kernel machines." In Advances in neural information processing systems, pp. 1177-1184. 2007.


X = dlmread('data/features_raw.dat', ' ');

n_features = size(X,2);
n_data = size(X,1);

gamma_inv = 0.1;
gamma = 1/gamma_inv;
sigma = sqrt(2/gamma_inv);
n_randomfeatures = 5040;
%calculate
W=normrnd(0,sigma,n_features,n_randomfeatures);
b=2*pi*rand(1,n_randomfeatures);
B = ones(n_data,1)*(b);
X_rrt = sqrt(2/n_randomfeatures)*cos(X*W+B);
disp('calculated');

%normalize
%mean_tr = mean(Data_new);
%std_tr = std(Data_new);
%Data_new = (Data_new-repmat(mean_tr,n_data,1))./(repmat(std_tr,n_data,1));
%disp('normalized');
%toc;

%write

dlmwrite('data/features_rrt.dat',X_rrt, ' ');

disp('writen to file');
