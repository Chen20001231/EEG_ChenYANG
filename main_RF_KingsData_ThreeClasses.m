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
addpath E:\Imperial\Spring\Project\GitKraken\EEG_ChenYANG_MakingDatasets\Local_Average_Reference\data_bipolar\
%addpath E:\Imperial\Spring\Project\GitKraken\EEG_ChenYANG_MakingDatasets\Local_Average_Reference\data_LAR\
%addpath E:\Imperial\Spring\Project\GitKraken\EEG_ChenYANG_MakingDatasets\Manually_Selecting_Testsets\DP141_2\data\
addpath functions\

fs = 250;
fs_new = 250;
num_of_channels = 30;
overlapping = 0.75;

%idx_testing_data_begin = 1;
%idx_testing_data_end = 47;

idx_testing_data_begin = 97;
idx_testing_data_end = 161;

%idx_testing_data_begin = 152;
%idx_testing_data_end = 168;
%% Start
counter = 1;
excel_table = readtable('0_segments.xlsx');
num_of_segments = height(excel_table);


%% Start to extract features

updateProgressBar(0);
for i = 1:num_of_segments
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
progressPercent = (i/num_of_segments)*40;
updateProgressBar(progressPercent);
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

y = string(excel_table.Category);
y = repelem(y, num_of_channels); % 将数组的每个元素重复 30 次



%% Partition data for cross-validation
% cv = cvpartition(length(y)/num_of_channels, 'HoldOut', 0.35);
% idxTrain = training(cv);
% extended_idxTrain = repelem(idxTrain, num_of_channels); % 将数组的每个元素重复 30 次

% Manual selection of training and test sets
idxTrain = ones(num_of_segments, 1);
% 前1-11为1
idxTrain(idx_testing_data_begin:idx_testing_data_end) = 0;
idxTrain = logical(idxTrain);

extended_idxTrain = repelem(idxTrain, num_of_channels); % 将数组的每个元素重复 30 次

x_train = x(extended_idxTrain,:);
y_train = y(extended_idxTrain,:);


%% testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
excel_table_testing = readtable('0_segments_testing.xlsx');
num_of_segments_testing = height(excel_table_testing);


counter = 1;
data=[];
feature=[];
for i = 1:num_of_segments_testing
    filename = ['y', num2str(i)];
    load(filename);
    [P,Q] = rat(fs_new/fs);
    for j = 1:num_of_channels
        data = EEGdata(:,j); % Channel
        data = resample(data,P,Q);
        feature(:,counter) = feature_extraction(data);
        counter = counter + 1;
    end
progressPercent = 40+(i/num_of_segments_testing)*50;
updateProgressBar(progressPercent);
end

x_test = feature';
y_test = string(excel_table_testing.Category);
y_test = repelem(y_test, num_of_channels); % 将数组的每个元素重复 30 次



%x_test = x(~extended_idxTrain,:);
%y_test = y(~extended_idxTrain,:);

idxTestOriginal = find(~extended_idxTrain);

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

%% feature importance
featureImportance = B.OOBPermutedPredictorDeltaError;
% 可视化特征重要性
figure;
bar(featureImportance);
xlabel('Feature Index');
ylabel('Out-of-Bag Permuted Predictor Delta Error');
title('Feature Importance');

%% 分segment统计
y_test_seg = y_test(1:num_of_channels:end);

grouped_data = reshape(y_pred, num_of_channels, []);  % 每一列代表一个组，共 30 列
counts = sum(strcmp(grouped_data, 'Seizure'));  % 统计每个组中 1 出现的次数
counts = [counts; sum(strcmp(grouped_data, 'NonSeizure'))];  % 统计每个组中 2 出现的次数
counts = [counts; sum(strcmp(grouped_data, 'PeriIctalSignals'))];  % 统计每个组中 3 出现的次数

%y_test_segNo = find(~idxTrain == 1);
y_test_segNo = (1:num_of_segments_testing)';
T3 = table(y_test_segNo, y_test_seg, counts(1,:)', counts(2,:)', counts(3,:)', 'VariableNames', {'Segment index','True value', '#ch pre as Seisure', '#ch pre as NonSeisure', '#ch pre as PeriIctalSignals'});
% 指定Excel文件的名称
filename3 = 'Incorrect_prediction_info_SegAsUnits_2.xlsx';
% 将表格写入Excel文件
writetable(T3, filename3);
% 显示完成信息
disp(['Data written to ', filename3]);


%% 画图
% 创建图形窗口
figure;
% 绘制第一个变量
subplot(2,1,1);
plot(y_test_segNo, counts(1,:), 'r:', 'LineWidth', 2);  % 红色实线
hold on;  % 保持当前图形
plot(y_test_segNo, counts(2,:), 'b:', 'LineWidth', 2); % 绿色虚线
plot(y_test_segNo, counts(3,:), 'm:', 'LineWidth', 2);  % 蓝色点线
xlim([min(y_test_segNo) max(y_test_segNo)]);
% 添加图例
legend('Seizure', 'NonSeizure', 'PeriIctalSignals');
% 添加坐标轴标签和标题
xlabel('Segment index');
ylabel('Number of channels');
title(' ');

grid on;
hold off;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
overlapping = 0.75;

idx_segment_plot_start = min(y_test_segNo);
idx_segment_plot_end = max(y_test_segNo);

EEGdataplot = [];
for q = idx_segment_plot_start:(1/(1-overlapping)):idx_segment_plot_end
    filename3 = ['y', num2str(q), '.mat'];
    load(filename3);
    EEGdataplot = [EEGdataplot; EEGdata];
end
set(gca,'linewidth',1,'fontsize',12,'fontname','Arial');

subplot(2,1,2);
% 定义偏移量，避免信号重叠
offset = 100;
% 遍历每个通道并绘制
hold on;
%for i = 24
for i = 1:num_of_channels
    plot(EEGdataplot(:, i) + (i-1) * offset);
end
hold off;

% 添加标签和标题
xlabel('Samples');
ylabel('Amplitude');
title([' ']);
grid on;

xlim([1,length(EEGdataplot)]);
ylim([-offset, (num_of_channels-1) * offset + offset]);
set(gca,'linewidth',1,'fontsize',12,'fontname','Arial');


%% Visualisation of error prediction by segment
%{
idx_segment_plot = 16;
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
%}
updateProgressBar(100);

