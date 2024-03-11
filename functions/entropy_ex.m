function [H] = entropy_ex(X)

X_uni = unique(X);
X_uni_size = numel(X_uni);
P = zeros(X_uni_size,1);
for i = 1:X_uni_size
    P(i) = sum(X == X_uni(i));
end
P = P ./ numel(X);
% Compute the Shannon's Entropy
H = -sum(P .* log2(P)); % 1.5

end

