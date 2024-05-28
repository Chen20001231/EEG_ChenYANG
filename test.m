clc;clear;close all;
num_of_channels = 30;

addpath E:\Imperial\Spring\Project\GitKraken\makeDatasets\data
addpath functions\
y1 = string(table2array(readtable('0_segments.xlsx','Range','C1:C42')));
y1 = repmat(y1, num_of_channels, 1);

y2 = string(table2array(readtable('0_segments.xlsx','Range','C42:C129')));
y2 = repmat(y2, num_of_channels, 1);