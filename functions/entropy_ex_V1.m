function [entropy_info] = entropy_ex_V1(data)
minVal = min(data);
maxVal = max(data);
data = 255*rescale(data, 'InputMin', minVal, 'InputMax', maxVal);
data = round(data);
entropy_info = entropy(uint8(data));

end

