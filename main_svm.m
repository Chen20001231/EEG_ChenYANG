clc;clear;close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data sampling rate of 32 kHz, down-sampled to 5 kHz
% In this project, down-sampled from 5 kHz to 128 Hz

% Wavelet
% cd1 2-4
% cd2 4-8
% cd3 8-16
% cd4 16-32
% cd5 32-64
% cd6 64-128

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% add path and parametre setting
addpath F:\eeg_mayo_data\DATASET_MAYO\
addpath functions\
fs = 5000;
fs_256 = 256;

%% Start
counter = 0;
for i = [42001:43000, 1:1000, 61001:62000, 118001:119000]
    counter = counter + 1;
    %% Load data
    filename = ['x', num2str(i-1,'%06d'), '.mat'];
    load(filename);

    %% change sampling frequency
    [P,Q] = rat(fs_256/fs);
    data = resample(data,P,Q);

    %% feature extraction
    feature(:,counter) = feature_extraction(data);

end


%% PCA

% Standardisation of data
for j = 1:10
    feature(j,:) = feature(j,:) - mean(feature(j,:));
    feature(j,:) = feature(j,:) ./ std(feature(j,:));
end

% Report covariance matrix, eigenvalues, and eigenvectors for the data.
covariance_matrix = cov(feature'); % covariance matrix
[eigen_vector, ~] = eig(covariance_matrix); % eigen vector and eigen value
e = eig(covariance_matrix);
[~,idx]=sort(e,'descend'); % Get the index of the eigenvalue magnitude


% Select Feature Vector for 1D projection
F1 = eigen_vector(:,idx(1));
F2 = eigen_vector(:,idx(2));
F3 = eigen_vector(:,idx(3));
F4 = eigen_vector(:,idx(4));
F5 = eigen_vector(:,idx(5));
% Get 1D data for PC1, PC2, and PC3
PC1 = feature'*F1;
PC2 = feature'*F2;
PC3 = feature'*F3;
PC4 = feature'*F4;
PC5 = feature'*F5;

% Create dataset
%x = [PC1, PC2, PC3, PC4, PC5];
x = [PC1, PC2];


%% add label

y=[ones(1000,1);2*ones(3000,1)];

%% SVM

%% Classification
KFold=5;
%% SVM
SVM = templateSVM('Standardize',1,'KernelFunction','gaussian');
indices = crossvalind('Kfold',y,KFold);Perfomance=[];
for i = 1:KFold
    test =(indices == i);train1 = ~test;
    TrainInputs=x(train1,:);TrainTargets=y(train1,:);
    TestInputs=x(test,:);labelTargetTest=y(test,:);
    Mdl = fitcecoc(TrainInputs,TrainTargets,'Learners',SVM,'FitPosterior',1,...
        'ClassNames',1:2,'Verbose',2);label = predict(Mdl,TestInputs);
    [Acc,Sen,Spe]=ConMax(labelTargetTest,label);Perfomance=[Perfomance;Acc Sen Spe]; %#ok   
end
clc;Perfomance=mean(Perfomance,1);disp('Svm: ACC SEN SPE')
disp(Perfomance)

%% KNN
Perfomance=[];indices = crossvalind('Kfold',y,KFold); 
for i = 1:KFold
   test =(indices == i);train1 = ~test;
    TrainInputs=x(train1,:);TrainTargets=y(train1,:);
    TestInputs=x(test,:);labelTargetTest=y(test,:);
    Mdl = fitcknn(TrainInputs,TrainTargets,'NumNeighbors',5,'Standardize',1);
    label = predict(Mdl,TestInputs);
    [Acc,Sen,Spe]=ConMax(labelTargetTest,label);
    Perfomance=[Perfomance;Acc Sen Spe]; %#ok
end
Perfomance=mean(Perfomance,1);disp('KNN: ACC SEN SPE');disp(Perfomance)
%% Naive Bayesian
Perfomance=[];indices = crossvalind('Kfold',y,KFold);
for i = 1:KFold
    test =(indices == i);train1 = ~test;
    TrainInputs=x(train1,:);TrainTargets=y(train1,:);
    TestInputs=x(test,:);labelTargetTest=y(test,:);
    CMdl=fitcnb(TrainInputs,TrainTargets);
    label = predict(CMdl,TestInputs);[Acc,Sen,Spe]=ConMax(labelTargetTest,label);
    Perfomance=[Perfomance;Acc Sen Spe]; %#ok
end
Perfomance=mean(Perfomance,1);disp('Naive Bayesian: ACC SEN SPE');disp(Perfomance)

