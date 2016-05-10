function y = curve_fitting(x, w, M)
% This function gives an output value when an input value polynomiais to form a polynomial function.
% e.g. y = w(0)*x^0 + w(1)*x^1 + w(2)*x^2 + ... + w(M)*x^(M+1)
%
% INPUT
% x: testing x (scalar)
% w: a vector of weight of which size is (M+1) by 1
% M: the number of order of polynomial 
%
% OUPUT
% y: w(0)*x^0 + w(1)*x^1 + w(2)*x^2 + ... + w(M)*x^(M+1)

phi = repmat(x,1, M+1);
y = 0;
for k = 0:M
    y = y + w(k+1) * (phi(k+1).^k);
end