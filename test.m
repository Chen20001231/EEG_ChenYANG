addpath functions\
% 总的迭代次数
total_iterations = 100;

% 开始循环
for i = 1:total_iterations
    % 进行一些计算（这里用 pause 模拟）
    pause(0.01); % 模拟一些计算时间
    
    % 计算当前进度百分比
    progressPercent = ((i-1) / total_iterations) * 100;
    
    % 更新进度条
    updateProgressBar(progressPercent);
    
end

% 最终完成进度条
updateProgressBar(100);