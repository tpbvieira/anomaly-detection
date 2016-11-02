function [output] = mp_density(x,gamma);
%
a_mp = (1-gamma^(-0.5))^2;
%
b_mp = (1+gamma^(-0.5))^2;
%
%if ((b_mp>x) & (a_mp<x) )
    output = gamma*(1/(2*pi*x))*sqrt((b_mp-x)*(x-a_mp));
%end