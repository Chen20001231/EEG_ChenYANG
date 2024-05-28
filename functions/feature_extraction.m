function [feature] = feature_extraction(data)

[cd1, cd2, cd3, cd4, cd5, cd6, cd7, cd8, ca1] = wavelet(data);
 
feature = zeros(2*nargin ,1);

feature(1) = energy_ex(cd1);
feature(2) = energy_ex(cd2);
feature(3) = energy_ex(cd3);
feature(4) = energy_ex(cd4);
feature(5) = energy_ex(cd5);
feature(6) = energy_ex(cd6);
feature(7) = energy_ex(cd7);
feature(8) = energy_ex(cd8);
feature(9) = energy_ex(ca1);

feature(10) = entropy_ex_V1(cd1);
feature(11) = entropy_ex_V1(cd2);
feature(12) = entropy_ex_V1(cd3);
feature(13) = entropy_ex_V1(cd4);
feature(14) = entropy_ex_V1(cd5);
feature(15) = entropy_ex_V1(cd6);
feature(16) = entropy_ex_V1(cd7);
feature(17) = entropy_ex_V1(cd8);
feature(18) = entropy_ex_V1(ca1);


feature(19) = std_ex(cd1);
feature(20) = std_ex(cd2);
feature(21) = std_ex(cd3);
feature(22) = std_ex(cd4);
feature(23) = std_ex(cd5);
feature(24) = std_ex(cd6);
feature(25) = std_ex(cd7);
feature(26) = std_ex(cd8);
feature(27) = std_ex(ca1);

feature(28) = kurtosis_ex(cd1);
feature(29) = kurtosis_ex(cd2);
feature(30) = kurtosis_ex(cd3);
feature(31) = kurtosis_ex(cd4);
feature(32) = kurtosis_ex(cd5);
feature(33) = kurtosis_ex(cd6);
feature(34) = kurtosis_ex(cd7);
feature(35) = kurtosis_ex(cd8);
feature(36) = kurtosis_ex(ca1);

feature(37) = skewness_ex(cd1);
feature(38) = skewness_ex(cd2);
feature(39) = skewness_ex(cd3);
feature(40) = skewness_ex(cd4);
feature(41) = skewness_ex(cd5);
feature(42) = skewness_ex(cd6);
feature(43) = skewness_ex(cd7);
feature(44) = skewness_ex(cd8);
feature(45) = skewness_ex(ca1);


end

