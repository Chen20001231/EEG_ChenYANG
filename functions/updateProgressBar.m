function updateProgressBar(progressPercent)
    % 检查输入是否在0到100之间
    if progressPercent < 0 || progressPercent > 100
        error('进度百分比必须在0到100之间');
    end

    progressBar=[];
    temp = length(progressBar);
    totalLength = 50; % 进度条总长度
    % 清除上一行的显示
    fprintf(repmat('\b', 1, temp));
    % 计算进度条长度
    completedLength = floor(progressPercent / 100 * totalLength);
    % 创建进度条字符串
    progressBar = ['Progress:','[',repmat('=', 1, completedLength),repmat(' ', 1, totalLength - completedLength),']'];
    % 使用 fprintf 打印进度条和百分比
    fprintf('%s %3d%%\n', progressBar, int8(progressPercent));
    
    % 当进度达到100%，结束进度条显示
    if progressPercent == 100
        fprintf('\n'); % 换行显示完成的进度条
    end
end