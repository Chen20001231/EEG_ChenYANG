%clc;clear;close all;
num_of_channels = 30;

addpath E:\Imperial\Spring\Project\GitKraken\makeDatasets\Three_classes\data\
addpath functions\

% 生成示例数据
data = randi([1, 3], 1, 1680);  % 生成含有 1680 个随机 1、2、3 的一维数组

% 将数据分成 30 个一组
grouped_data = reshape(data, 30, []);  % 每一列代表一个组，共 30 列

% 统计每个组中 1、2、3 出现的次数
counts = sum(grouped_data == 1);  % 统计每个组中 1 出现的次数
counts = [counts; sum(grouped_data == 2)];  % 统计每个组中 2 出现的次数
counts = [counts; sum(grouped_data == 3)];  % 统计每个组中 3 出现的次数

% 显示结果
disp(counts);