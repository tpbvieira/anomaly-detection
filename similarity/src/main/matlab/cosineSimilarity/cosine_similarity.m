%% Cosine similarity and vector norms
% Given two vectors, one measure of their similarity is the cosine of the
% angle between the vector.
% Suppose that $a$ and $b$ are both vectors.

%% Download the Yale dataset
% Deng Cai at Zhejiang university made some of the common image databases
% available in Matlab format at his webpage: 
%   http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html
% Thanks!  We are going to download one.
urlwrite('http://www.cad.zju.edu.cn/home/dengcai/Data/Yale/Yale_32x32.mat','Yale_32x32.mat');

%% 
% Cool, that went out and downloaded the file right from Matlab!  No 
% Web browser needed.

%% Load the Yale dataset
load Yale_32x32

%% Showing an image
% The matrix fea is an image where each row is an image.
I = reshape(fea(1,:),32,32);
imagesc(I); colormap(gray);

%% Let's find the closest face to this image using cosine similarity.
% This will involve transforming the data, slightly...

%% 
% First, take the transpose of the data. Now, each column is an image
A = fea';

%%
% Second, normalize the columns
n = size(A,2); % n is the number of images, which is the number of columns
for i=1:n
    A(:,i) = A(:,i)/norm(A(:,i)); % set the ith column of A to be the 
                                  % ith column of A, normalized.
end

%%
% Third, compute cosine similarities against the first image
z = fea(1,:)';  % get the first image
z = z/norm(z);  % normalize it
s = A'*z;       % compute all the cosines!

%%
% The cosine of an angle in the same orthant is between 0 and 1, let's check this!
plot(s)

%% 
% Let's find the closest image
[ignore p] = sort(s,'descend'); % sort s in descending order and 
                                % store the sorting permutation in p
% p(1) is the closest image to the first one.
p(1)

%%
% Hey, it's closest to itself!  What's the next closest?
p(2)

%%
% Let's see it!
subplot(1,2,1);
imagesc(reshape(fea(p(1),:),32,32)); colormap(gray);
title('original image');
subplot(1,2,2);
imagesc(reshape(fea(p(10),:),32,32)); colormap(gray);
title('closest image');