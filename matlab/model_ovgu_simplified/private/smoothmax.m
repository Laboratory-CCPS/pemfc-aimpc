function [out] = smoothmax(x,y,a)
%smoothmax(x,y,a) calculates a smooth approximation of the maximum function
%  for two arguments x and y, with scaling parameter a

% Boltzmann operator
% out = (y*exp(a*y)+x.*exp(a*x))./(exp(a*y)+exp(a*x));

% Mellowmax
% out = 1/a*log( 0.5*(exp(a*x)+exp(a*y)) );

% LogSumExp
out = 1/a*log( (exp(a*x)+exp(a*y)) );

% Smooth maximum unit
% out = (x+y +((x-y).^2+1/a).^0.5)/2;

end
