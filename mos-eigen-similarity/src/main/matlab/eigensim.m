function y = eigensim(X,w,method)
% eigensim 
%
% SYNOPSIS: 
%
% INPUT X: data matrix with columns as observations and lines as variables
%       w: window size
%       l: number of windows for attack detection
%       method: 'unit' for zero mean and unit variance or 'norm' for
%       normalization by zero mean
%
% OUTPUT y: 
%
% EXAMPLE eigensim(X,w,method)
%
% SEE ALSO 
%
% created with MATLAB R2016a on Ubuntu 16.04
% created by: Thiago Vieira
% DATE: 01-Oct-2010
%

y = zeros(size(X,2))