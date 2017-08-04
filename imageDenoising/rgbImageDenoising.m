clc;                                                                        % Clear command window.
clear;                                                                      % Delete all variables.
close all;                                                                  % Close all figure windows except those created by imtool.
imtool close all;                                                           % Close all figure windows created by imtool.

workspace;                                                                  % Make sure the workspace panel is showing.
fontSize = 15;

% Read in a standard MATLAB color demo image.
folder = fullfile(matlabroot, '\toolbox\images\imdemos');
baseFileName = 'peppers.png';
fullFileName = fullfile(folder, baseFileName);
% Get the full filename, with path prepended.
fullFileName = fullfile(folder, baseFileName);
if ~exist(fullFileName, 'file')
	% Didn't find it there.  Check the search path for it.
	fullFileName = baseFileName; % No path this time.
	if ~exist(fullFileName, 'file')
		% Still didn't find it.  Alert user.
		errorMessage = sprintf('Error: %s does not exist.', fullFileName);
		uiwait(warndlg(errorMessage));
		return;
	end
end
rgbImage = imread(fullFileName);

% Get the dimensions of the image.  numberOfColorBands should be = 3.
[rows columns numberOfColorBands] = size(rgbImage);

% Display the original color image.
subplot(3, 4, 1);
imshow(rgbImage);
title('Original color Image', 'FontSize', fontSize);

% Enlarge figure to full screen.
set(gcf, 'Position', get(0,'Screensize')); 

% Extract the individual red, green, and blue color channels.
redChannel = rgbImage(:, :, 1);
greenChannel = rgbImage(:, :, 2);
blueChannel = rgbImage(:, :, 3);

% Display the individual red, green, and blue color channels.
subplot(3, 4, 2);
imshow(redChannel);
title('Red Channel', 'FontSize', fontSize);
subplot(3, 4, 3);
imshow(greenChannel);
title('Green Channel', 'FontSize', fontSize);
subplot(3, 4, 4);
imshow(blueChannel);
title('Blue Channel', 'FontSize', fontSize);

% Generate a noisy image.  This has salt and pepper noise independently on
% each color channel so the noise may be colored.
noisyRGB = imnoise(rgbImage,'salt & pepper', 0.05);
subplot(3, 4, 5);
imshow(noisyRGB);
title('Image with Salt and Pepper Noise', 'FontSize', fontSize);

% Extract the individual red, green, and blue color channels.
redChannel = noisyRGB(:, :, 1);
greenChannel = noisyRGB(:, :, 2);
blueChannel = noisyRGB(:, :, 3);

% Display the noisy channel images.
subplot(3, 4, 6);
imshow(redChannel);
title('Noisy Red Channel', 'FontSize', fontSize);
subplot(3, 4, 7);
imshow(greenChannel);
title('Noisy Green Channel', 'FontSize', fontSize);
subplot(3, 4, 8);
imshow(blueChannel);
title('Noisy Blue Channel', 'FontSize', fontSize);

% Median Filter the channels:
redMF = medfilt2(redChannel, [3 3]);
greenMF = medfilt2(greenChannel, [3 3]);
blueMF = medfilt2(blueChannel, [3 3]);

% Find the noise in the red.
noiseImage = (redChannel == 0 | redChannel == 255);

% Get rid of the noise in the red by replacing with median.
noiseFreeRed = redChannel;
noiseFreeRed(noiseImage) = redMF(noiseImage);

% Find the noise in the green.
noiseImage = (greenChannel == 0 | greenChannel == 255);

% Get rid of the noise in the green by replacing with median.
noiseFreeGreen = greenChannel;
noiseFreeGreen(noiseImage) = greenMF(noiseImage);

% Find the noise in the blue.
noiseImage = (blueChannel == 0 | blueChannel == 255);

% Get rid of the noise in the blue by replacing with median.
noiseFreeBlue = blueChannel;
noiseFreeBlue(noiseImage) = blueMF(noiseImage);

% Reconstruct the noise free RGB image
rgbFixed = cat(3, noiseFreeRed, noiseFreeGreen, noiseFreeBlue);
subplot(3, 4, 9);
imshow(rgbFixed);
title('Restored Image', 'FontSize', fontSize);