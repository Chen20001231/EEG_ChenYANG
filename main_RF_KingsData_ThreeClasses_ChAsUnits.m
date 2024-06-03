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
addpath E:\Imperial\Spring\Project\GitKraken\EEG_ChenYANG_MakingDatasets\Three_classes\data
addpath functions\
fs = 250;
fs_new = 250;
num_of_channels = 30;

%% Start
counter = 1;
for i = 1:162
    %% Load data
    filename = ['x', num2str(i), '.mat'];
    load(filename);

    %% change sampling frequency
    [P,Q] = rat(fs_new/fs);

    for j = 1:num_of_channels
        data = EEGdata(:,j); % Channel
        data = resample(data,P,Q);
        %% feature extraction
        feature(:,counter) = feature_extraction(data);
        counter = counter + 1;
    end


end
%{
[cd1, cd2, cd3, cd4, cd5, cd6, cd7, cd8, ca1] = wavelet(data);
minVal = min(ca1);
maxVal = max(ca1);
ca1 = 255*rescale(ca1, 'InputMin', minVal, 'InputMax', maxVal);
ca1 = round(ca1);
%}
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

y1 = string(table2array(readtable('0_segments.xlsx','Range','C1:C42')));
y1 = repmat(y1, num_of_channels, 1);
y2 = string(table2array(readtable('0_segments.xlsx','Range','C42:C129')));
y2 = repmat(y2, num_of_channels, 1);
y3 = string(table2array(readtable('0_segments.xlsx','Range','C129:C163')));
y3 = repmat(y3, num_of_channels, 1);

y = [y1;y2;y3];
%data_labeled = [x, y];

%% Partition data for cross-validation
cv = cvpartition(length(y), 'HoldOut', 0.35);
idxTrain = training(cv);
x_train = x(idxTrain,:);
y_train = y(idxTrain,:);
x_test = x(~idxTrain,:);
y_test = y(~idxTrain,:);

idxTestOriginal = find(~idxTrain);

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
numTrees = 50; % Set number of trees
opts = statset('UseParallel',true); % Parallel computing
    
% Use decision trees
B = TreeBagger(numTrees, x_train, y_train, 'Method', 'classification', 'Options', opts,'OOBPredictorImportance', 'on');
% B = TreeBagger(numTrees, x_train, y_train, 'Method', 'classification', 'Options', opts, 'MaxNumSplits', 5);

% Predicted data
y_pred = predict(B, x_test);

view(B.Trees{1}, 'Mode', 'graph');
view(B.Trees{2}, 'Mode', 'graph');


%% feature importance
featureImportance = B.OOBPermutedPredictorDeltaError;
% 可视化特征重要性
figure;
bar(featureImportance);
xlabel('Feature Index');
ylabel('Out-of-Bag Permuted Predictor Delta Error');
title('Feature Importance');

%% Display a confusion matrix and comment on the overall accuracy.
C = confusionmat(y_test, y_pred);
order = {'Seizure','NonSeizure','PreSeizure'};

% Display a confusion matrix 
figure;
cm = confusionchart(C,order);
cm.ColumnSummary = 'column-normalized';
title('Confusion Matrix');
xlabel('Predicted Label');
ylabel('True Label');

% Displat the overall accuracy
for i = 1:length(y_test)
    if strcmp(y_test(i), 'Seizure') == 1
        y_test_temp(i)=1;
    elseif strcmp(y_test(i), 'NonSeizure') == 1
        y_test_temp(i)=2;
    else
        y_test_temp(i)=3;
    end
end

for i = 1:length(y_pred)
    if strcmp(y_pred(i), 'Seizure') == 1
        y_pred_temp(i)=1;
    elseif strcmp(y_pred(i), 'NonSeizure') == 1
        y_pred_temp(i)=2;
    else
        y_pred_temp(i)=3;
    end
end


accuracy = sum(strcmp(y_test, y_pred)) / numel(y_test);
disp(['Overall accuracy: ', num2str(accuracy)]);
disp('----------------');

incorrect_indices = find(~strcmp(y_test, y_pred));

idx_segment=[];
idx_channel=[];
for n=1:length(incorrect_indices)
    original_incorrect_indices(n)=idxTestOriginal(incorrect_indices(n))-1;
    idx_segment(n) = ceil(original_incorrect_indices(n) / num_of_channels);
    idx_channel(n) = mod(original_incorrect_indices(n), num_of_channels)+1;
    signal_true(n) = y_test(incorrect_indices(n));
    signal_predicted(n) = y_pred(incorrect_indices(n));
end

T = table((1:length(incorrect_indices))', idx_segment', idx_channel', signal_true', signal_predicted', 'VariableNames', {'Index', 'segment', 'channel', 'true value', 'predicted value'});
% 指定Excel文件的名称
filename2 = 'Incorrect_prediction_info.xlsx';
% 将表格写入Excel文件
writetable(T, filename2);
% 显示完成信息
disp(['Data written to ', filename2]);

%% Visualisation of error prediction by single channel

counter = 1;
for k = 1:length(incorrect_indices)
    idx_original_incorrect(k)=idxTestOriginal(incorrect_indices(k))-1;
    idx_segment_temp(k) = ceil(idx_original_incorrect(k) / num_of_channels);
    idx_channel_temp(k) = mod(idx_original_incorrect(k), num_of_channels)+1;
    signal_true_temp(k) = y_test(incorrect_indices(k));
    signal_predicted_temp(k) = y_pred(incorrect_indices(k));
    filename3 = ['x', num2str(idx_segment_temp(k)), '.mat'];
    load(filename3);
    data_plot = EEGdata(:,idx_channel_temp(k)); % Channel
    figure('Visible', 'off');
    plot(data_plot,'b');
    hold on;
    xlabel('Samples');
    ylabel('Amplitude');
    title(['EEG Signals from segment. ', num2str(idx_segment_temp(k)),', channel.',num2str(idx_channel_temp(k))]);
    % 获取当前轴的限制
    xLimits = xlim;
    yLimits = ylim;0
    % 定义文字的位置（绘图的左上角）
    xPos = xLimits(1) + 0.05 * (xLimits(2) - xLimits(1)); % x 坐标，距离左边界 5%
    yPos = yLimits(2) - 0.05 * (yLimits(2) - yLimits(1)); % y 坐标，距离上边界 5%
    % 添加文字
    str_temp = strcat(signal_true_temp(k),'>>',signal_predicted_temp(k));
    text(xPos, yPos, str_temp, 'Color', 'black', 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', 'white');
    grid on;
    hold off;
    % 定义保存路径和文件名
    filePath = string(['E:\Imperial\Spring\Project\GitKraken\fig_incorrect\',num2str(counter),'.png']);
    counter = counter + 1;
    % 保存绘图到指定路径
    saveas(gcf, filePath);
end

%% Visualisation of error prediction by segment
idx_segment_plot = 36;
filename3 = ['x', num2str(idx_segment_plot), '.mat'];
load(filename3);

figure;
% 定义偏移量，避免信号重叠
offset = 100;
% 遍历每个通道并绘制
hold on;
%for i = 24
for i = 1:num_of_channels
    plot(EEGdata(:, i) + (i-1) * offset);
end
hold off;

% 添加标签和标题
xlabel('Samples');
ylabel('Amplitude');
title(['EEG Signals from segment. ', num2str(idx_segment_plot)]);
grid on;
ylim([-offset, (num_of_channels-1) * offset + offset]);




