function D = get64x256sine()
% get64x256sine   Get a 64x256 separable dictionary with sine elements
%
% D = get64x256sine();

D = zeros(64,256);
F = getNxKsine(8, 16, 60, 300);

i = 1;
for j=1:16
    for k=1:16
        D(:,i) = reshape(F(:,j)*F(:,k)',64,1);
        i = i+1;
    end
end

return