function D = get8x21haar()
% get8x21haar     Get the 8x21 regular Haar dictionary (frame)
%
% D = get8x21haar();

a = sqrt(1/8);
b = 1/2;
c = sqrt(1/2);
D = zeros(8,21);
D(:,1) = a*ones(8,1);
x = [a,a,a,a,-a,-a,-a,-a]';
for i = 2:5
    D(:,i) = x;
    x = [x(8);x(1:7)];
end
x = [b,b,-b,-b,0,0,0,0]';
for i = 6:13
    D(:,i) = x;
    x = [x(8);x(1:7)];
end
x = [c,-c,0,0,0,0,0,0]';
for i = 14:21
    D(:,i) = x;
    x = [x(8);x(1:7)];
end

return;

