clc;clear;close all;
num_of_channels = 30;

addpath E:\Imperial\Spring\Project\GitKraken\makeDatasets\Three_classes\data\
addpath functions\

% 定义一个已知数组
array = [1, 2, 3, 4, 5];

% 使用 for 循环遍历数组的每个元素
for k = array
    % 在此处添加你的代码，使用 k 进行操作
    disp(['当前 k 的值是: ', num2str(k)]);
end