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
% cd7 128-256
% cd8 256-512

% ca8 512-1024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% add path and parametre setting
addpath F:\eeg_mayo_data\DATASET_MAYO\
addpath functions\
fs = 5000;
fs_new = 1024;

%% Start
counter = 0;
for i = [1:1000, 42001:43000, 61001:62000, 118001:119000]
    counter = counter + 1;
    %% Load data
    filename = ['x', num2str(i-1,'%06d'), '.mat'];
    load(filename);

    %% change sampling frequency
    [P,Q] = rat(fs_new/fs);
    data = resample(data,P,Q);

    %% feature extraction
    feature(:,counter) = feature_withoutWL_extraction(data);

end


%% PCA
%{
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
x = [PC1, PC2, PC3, PC4, PC5];
%}
x = feature';

%% add label

y1 = string(table2array(readtable('segments.csv','Range','K2:K1001')));
y2 = string(table2array(readtable('segments.csv','Range','K42002:K43001')));
y3 = string(table2array(readtable('segments.csv','Range','K61002:K62001')));
y4 = string(table2array(readtable('segments.csv','Range','M2:M1001')));

y = [y1;y2;y3;y4];
%data_labeled = [x, y];

%% Partition data for cross-validation
cv = cvpartition(length(y), 'HoldOut', 0.4);
idxTrain = training(cv);
x_train = x(idxTrain,:);
y_train = y(idxTrain,:);
x_test = x(~idxTrain,:);
y_test = y(~idxTrain,:);


%% Number of decision trees
%{
for i = 1:50
    % Define Bagging Parameters
    numTrees = i; % Set number of trees
    opts = statset('UseParallel',true); % Parallel computing
        
    % Use decision trees
    B = TreeBagger(numTrees, x_train, y_train, 'Method', 'classification', 'Options', opts);
    %B = TreeBagger(numTrees, x_train, y_train, 'Method', 'classification', 'Options', opts, 'MaxNumSplits', 8);
    
    % Predicted data
    y_pred = predict(B, x_test);
    err(i) = 1-sum(strcmp(y_test, y_pred)) / numel(y_test);

end

figure();
plot(err, 'b-','LineWidth',1);
%title('Scree plot');
xlabel('Trees Grown','Fontname', 'Arial','FontSize',12);
ylabel('Error','Fontname', 'Arial','FontSize',12);
set(gca,'linewidth',1,'fontsize',12,'fontname','Arial');
grid on;
%}

%% Visualise two of the generated decision trees. 
% Define Bagging Parameters
numTrees = 30; % Set number of trees
opts = statset('UseParallel',true); % Parallel computing
    
% Use decision trees
B = TreeBagger(numTrees, x_train, y_train, 'Method', 'classification', 'Options', opts,'OOBPredictorImportance','on');
% B = TreeBagger(numTrees, x_train, y_train, 'Method', 'classification', 'Options', opts, 'MaxNumSplits', 5);

figure();
featureImportance = B.OOBPermutedVarDeltaError; % Feature importance
bar(featureImportance);
xlabel('Feature index');
ylabel('Out-of-Bag Feature Importance');
title('Feature importance');

% Predicted data
y_pred = predict(B, x_test);

view(B.Trees{1}, 'Mode', 'graph');
view(B.Trees{2}, 'Mode', 'graph');

%% Display a confusion matrix and comment on the overall accuracy.
C = confusionmat(y_test, y_pred);
order = {'noise','pathology','physiology','powerline'};

% Display a confusion matrix 
figure;
cm = confusionchart(C,order);
cm.ColumnSummary = 'column-normalized';
title('Confusion Matrix');
xlabel('Predicted Label');
ylabel('True Label');

% Displat the overall accuracy
for i = 1:length(y_test)
    if strcmp(y_test(i), 'pathology') == 1
        y_test_temp(i)=1;
    else
        y_test_temp(i)=2;
    end
end

for i = 1:length(y_pred)
    if strcmp(y_pred(i), 'pathology') == 1
        y_pred_temp(i)=1;
    else
        y_pred_temp(i)=2;
    end
end

[Acc,Sen,Spe]=ConMax(y_test_temp,y_pred_temp);
Perfomance=[Acc Sen Spe];
disp('Accuracy Sensitivity Specificity');
disp(Perfomance)
disp('----------------');

accuracy = sum(strcmp(y_test, y_pred)) / numel(y_test);
disp(['Overall accuracy: ', num2str(accuracy)]);
disp('----------------');

