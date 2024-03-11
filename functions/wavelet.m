function [cd1, cd2, cd3, cd4, cd5, cd6, cd7, cd8, ca8] = wavelet(data)
%% design wavelet

wname='db4';
nLevel=8;
%[a,d]=swt(data,nLevel,wname); % wavelet method 2

[C,L]=wavedec(data,nLevel,wname);
[cd1, cd2, cd3, cd4, cd5, cd6, cd7, cd8] = detcoef(C,L,[1,2,3,4,5,6,7,8]);
ca8 = appcoef(C,L,wname,nLevel); %freq 512-1024

%{
figure;
subplot(5,1,1);
plot(cd1);
title('cd1');

subplot(5,1,2);
plot(cd2);
title('cd2');

subplot(5,1,3);
plot(cd3);
title('cd3');

subplot(5,1,4);
plot(cd4);
title('cd4');

subplot(5,1,5);
plot(cd5);
title('cd5');
%}

end

