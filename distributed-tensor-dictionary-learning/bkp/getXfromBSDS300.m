function X = getXfromBSDS300(varargin)
% getXfromBSDS300 Get (training) vectors generated from BSDS300 images
%                 This is the Berkeley Segmentation Dataset from 
% http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/
% 
% The images are 481x321 RGB color images stored as JPEG files, '*.jpg'. 
% A RGB image can be transformed to YCbCr and only the Y component is used
% when a grayscale image is requested.
% The vectors are not normalized to unit 2-norm, 
% the mean for each color may or may not be subtracted. 
% To normalize do:
%    X = getXfromBSDS300('sm',1, 'norm1lim',0.1, argName, argVal, ...);
%    X = X ./ (ones(size(X,1),1)*sqrt(sum(X.*X))); 
% 
% see also: getXfrom8images, myim2col
%
% X = getXfromBSDS300(argName, argVal, ...);
%----------------------------------------------------------------------
%        The input arguments (options) may be:
%  'col' is 1 (true) or 0 (false), default is 0.
%  'bw' is 1 (true) or 0 (false), default is 1
%  'transform' or 't' is as in myim2col, default is 'none',
%        Other values are: 'dct', 'lot', 'elt', 'db79', 'm79', ...
%  'myim2colPar' more parameters to use in myim2col, default: 
%        struct('s',[8,8], 'n',[8,8], 'i',[8,8], 'a','none', 'v',0);
%        'a','none' : mean that images are cut off, trunc, to 480x320
%        'n' gives the size of the neighborhood and must be included
%        'i',[32,32] find max floor(321/32)*floor(481/32) = 150 patches 
%        and min (10-2)*(15-2) = 104 patches from each image.      
%        (depending on random offset)
%  'noTotal' or 'no' how many training vectors to get 
%  'subtractMean' or 'sm' is 1 (true) or 0 (false), default is 0
%       Mean is subtacted for each vector, if RGB is used mean of each
%       component is subtracted (i.e. thre values for each vector).
%  'norm1lim' only vectors with |x|_1 >= norm1lim are used, default 0
%       when used flat or nearly flat patches will be excluded. This is
%       normally only used in combination with 'subtractMean' == true
%  'images', a cell of strings with the filenames (images). 
%        or an array with numbers indicating the images to use from
%        training (198 images) or test catalog (100 images).
%        Default is 10 random images from catalog 'imCat'
%  'imCat', a cell of strings with the catalog names where the images
%        may be located. All used images are the same catalog, and the
%        first catalog where the first JPEG-image is found is used. 
%        defalt is {pwd}  (present work directory)
%        SPECIAL cases when imCat is a string (char array)
%        'train' or 'tr' set imCat to {my_matlab_path('BSDStrain')} 
%        'test' or 'te' set imCat to {my_matlab_path('BSDStest')}
%  'verbose' or 'v' to indicate verboseness, default 0
%----------------------------------------------------------------------
% example: 
% par = struct('imCat','train', 'no',1000, 'col',false, 'subtractMean',true, 'norm1lim',10, ...
%            'myim2colPar',struct('n',[12,12], 'i',[12,12], 'a','none', 'v',0) );
% parA = struct('imCat','train', 'no',1000, 'col',false, 'subtractMean',true, 'norm1lim',10, ...
%            'myim2colPar',struct('n',[8,8], 'i',[32,32], 'a','none', 'v',0) );
% parAt = struct('imCat','train', 'no',1000, 'col',false, 'subtractMean',true, 'norm1lim',10, ...
%            'transform','dct', 'myim2colPar',struct('n',[8,8], 'i',[32,32], 'a','none', 't','dct', 'v',0) );
% parB = struct('imCat','train', 'no',1000, 'col',true, 'subtractMean',true, 'norm1lim',10, ...
%            'myim2colPar',struct('n',[12,12], 'i',[32,32], 'a','none', 'v',0) );
% parC = struct('imCat','train', 'no',1000, 'col',false, 'subtractMean',true, 'norm1lim',10, ...
%            'myim2colPar',struct('n',[16,16], 'i',[32,32], 'a','none', 'v',0) );
% X = getXfromBSDS300(par, 'v',1);   % from 10 random images
% Xa = getXfromBSDS300(parA, 'v',1, 'images',[1:5,45:49]);  % some specified images
% Xb = getXfromBSDS300(parB, 'v',1);  
% Xc = getXfromBSDS300(parC, 'v',1);  

%----------------------------------------------------------------------
% Copyright (c) 2010.  Karl Skretting.  All rights reserved.
% University of Stavanger (Stavanger University), Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.his.no/~karlsk/
% 
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  23.12.2010  Made function 
% Ver. 1.1  10.01.2013  may use my_matlab_path(..) to set imCat
%----------------------------------------------------------------------

%% default options
useBW = true;
transform = 'none';       
myim2colPar = struct('s',[8,8], 'n',[8,8], 'i',[8,8], 'a','none', 't','none', 'v',0);
noTotal = 100;
subtractMean = false;
norm1lim = 0;
images = 0;   % indicate all JPEG images in first catalog
imCat = {pwd, my_matlab_path('BSDStest')};  
verbose = 0;
if isunix()
    catSep = '/';
else
    catSep = '\';
end

%%  get the options
nofOptions = nargin;
optionNumber = 1;
fieldNumber = 1;
while (optionNumber <= nofOptions)
    if isstruct(varargin{optionNumber})
        sOptions = varargin{optionNumber}; 
        sNames = fieldnames(sOptions);
        opName = sNames{fieldNumber};
        opVal = sOptions.(opName);
        % next option is next field or next (pair of) arguments
        fieldNumber = fieldNumber + 1;  % next field
        if (fieldNumber > numel(sNames)) 
            fieldNumber = 1;
            optionNumber = optionNumber + 1;  % next pair of options
        end
    elseif iscell(varargin{optionNumber})
        sOptions = varargin{optionNumber}; 
        opName = sOptions{fieldNumber};
        opVal = sOptions{fieldNumber+1};
        % next option is next pair in cell or next (pair of) arguments
        fieldNumber = fieldNumber + 2;  % next pair in cell
        if (fieldNumber > numel(sOptions)) 
            fieldNumber = 1;
            optionNumber = optionNumber + 1;  % next pair of options
        end
    else
        opName = varargin{optionNumber};
        opVal = varargin{optionNumber+1};
        optionNumber = optionNumber + 2;  % next pair of options
    end
    % interpret opName and opVal
    if strcmpi(opName,'col') 
        if islogical(opVal); useBW = ~opVal; end;
        if isnumeric(opVal); useBW = (opVal(1)==0); end;
    end
    if strcmpi(opName,'bw') 
        if islogical(opVal); useBW = opVal; end;
        if isnumeric(opVal); useBW = (opVal(1)~=0); end;
    end
    if strcmpi(opName,'subtractMean') || strcmpi(opName,'sm') 
        if islogical(opVal); subtractMean = opVal; end;
        if isnumeric(opVal); subtractMean = (opVal(1)~=0); end;
    end
    if strcmpi(opName,'norm1lim') && isnumeric(opVal)
        norm1lim = opVal(1); 
    end
    %
    if (strcmpi(opName,'transform') || strcmpi(opName,'t'))
        transform = opVal;
    end
    if strcmpi(opName,'myim2colPar') 
        if (iscell(opVal) || isstruct(opVal))
            myim2colPar = opVal;
        else
            error('getXfromBSDS300: illegal option for myim2colPar, it is ignored.');
        end
    end
    if ( (strcmpi(opName,'noTotal') || strcmpi(opName,'no')) && isnumeric(opVal) )
        noTotal = opVal(1);
    end
    if ( (strcmpi(opName,'images')) && isnumeric(opVal) )
        images = opVal(:)';     % row vector
    end
    if ( (strcmpi(opName,'images')) && iscell(opVal) )
        images = opVal;    
    end
    if ( (strcmpi(opName,'images')) && ischar(opVal) )
        images = cell(1,size(opVal,1));
        for i=1:numel(images)
            images{i} = strtrim(opVal(i,:));
        end
    end
    if  strcmpi(opName,'imCat')
        if iscell(opVal)
            imCat = opVal;
        elseif (ischar(opVal) && (numel(opVal) >= 2))
            if strcmpi(opVal(1:2),'te')
                imCat = {my_matlab_path('BSDStest')};
            end
            if strcmpi(opVal(1:2),'tr')
                imCat = {my_matlab_path('BSDStrain')};
            end
        end
    end
    if strcmpi(opName,'verbose') || strcmpi(opName,'v')
        if (islogical(opVal) && opVal); verbose = 1; end;
        if isnumeric(opVal); verbose = opVal(1); end;
    end
end
myim2colPar.t = transform;

%% check where to find the images
for i=1:numel(imCat)
    catalog = imCat{i};
    if isnumeric(images)
        dirFiles = dir([catalog,catSep,'*.jpg']);
        if (numel(dirFiles) > 0)
            if (images(1) == 0)
                images = randperm(numel(dirFiles)); 
                if (numel(images) >= 10)
                    images = images(1:10);  % 10 random images
                end
            else
                images = unique(max( rem(floor(images)-1,numel(dirFiles))+1, 1 ));
            end
            break; 
        end
    elseif iscell(images)
        if exist([catalog,catSep,images{1}],'file'); break; end
    end
end
if iscell(images)
    for i=1:numel(images)
        if ~exist([catalog,catSep,images{i}],'file'); 
            error(['getXfromBSDS300: did not find ',catalog,images{i},'.']);
        end
    end
end
% 'catalog' and 'images' ok    
if norm1lim > 0
    noFromEachIm = ceil( 1.05*noTotal/numel(images) );
else
    noFromEachIm = ceil( noTotal/numel(images) );
end
%
if verbose
    disp(['getXfromBSDS300: Use ',int2str(numel(images)),' images from ',catalog]);
end

%% make the training/test data
L = numel(images)*noFromEachIm;   
if useBW
    X = zeros(prod(myim2colPar.n), L);
else
    X = zeros(3*prod(myim2colPar.n), L);
end
if exist('ME','var'); clear('ME'); end;   
for i = 1:numel(images)
    if isnumeric(images)
        imFilename = [catalog,catSep,dirFiles(images(i)).name];
    else
        imFilename = [catalog,catSep,images{i}];
    end
    try
        A = double( imread(imFilename) );  % do not subtract global 'mean'
        if ~useBW && (size(A,3) < 3)
            ME = struct('message',['Color image is expected but ',imFilename,' is not.']);
            disp(['getXfromBSDS300: skip image ',imFilename]);
            disp(['Warning: ',ME.message]);
            continue;
        end
        if useBW && (size(A,3) > 1)  
            if  (size(A,3) == 3)  % assume file is RGB colorspace
                % A = rgb2ycbcr(A); A = A(:,:,1); 
                A = 0.299*A(:,:,1) + 0.587*A(:,:,2) + 0.114*A(:,:,3);
            end
            A = A(:,:,1);  % use first 'color'
        end
        % problem med imOffset i myim2col for transformer ??
        imOffset = floor( myim2colPar.n .* rand(1,2) );
        % imOffset = [0,0];
        if (size(A,3) == 1)
            Xi = myim2col( A, myim2colPar, 'offset',imOffset );
            if subtractMean
                if strcmpi(transform,'none')
                    Xi = Xi - repmat(mean(Xi), size(Xi,1), 1);
                else
                    Xi(1,:) = 0;
                end
            end
        elseif (size(A,3) == 3)
            X1 = myim2col( A(:,:,1), myim2colPar, 'offset',imOffset );
            X2 = myim2col( A(:,:,2), myim2colPar, 'offset',imOffset );
            X3 = myim2col( A(:,:,3), myim2colPar, 'offset',imOffset );
            if subtractMean 
                if strcmpi(transform,'none')
                    X1 = X1 - repmat(mean(X1), size(X1,1), 1); 
                    X2 = X2 - repmat(mean(X2), size(X2,1), 1); 
                    X3 = X3 - repmat(mean(X3), size(X3,1), 1); 
                else
                    X1(1,:) = 0;
                    X2(1,:) = 0;
                    X3(1,:) = 0;
                end
            end
            Xi = [X1; X2; X3];
        end
        I = randperm(size(Xi,2));
        if (size(Xi,2) <= noFromEachIm)
            top = i + (size(Xi,2)-1)*numel(images);
            X(:, i:numel(images):top ) = Xi(:,I);  
            t = ['  find ',int2str(size(Xi,2)),' patches from ',imFilename, ...
                '  (but ',int2str(noFromEachIm),' was wanted)'];
            ME = struct('message',t);  % a 'normal' exception is indicated but not thrown
            if verbose
                disp(ME.message);  
            end
        else
            X(:, i:numel(images):L ) = Xi(:, I(1:noFromEachIm) );   
            t = ['  find ',int2str(noFromEachIm),' patches from ',imFilename];
            if verbose
                disp(t);  
            end
        end
    catch ME
        disp(['getXfromBSDS300: skip image ',imFilename]);
        disp(['An error occurred: ',ME.message]);
    end
end
if exist('ME','var')     % if an exception occured
    X = X(:,sum(abs(X))~=0);
end
if norm1lim > 0
    X = X(:,sum(abs(X)) >= norm1lim);
end

I = randperm(size(X,2));
if size(X,2) < noTotal
    X = X(:, I ); 
    disp(['  WARNING: return only ',int2str(size(X,2)),' vectors in X.']);
else
    X = X(:, I(1:noTotal) ); 
    if verbose
        disp(['  and return ',int2str(noTotal),' of these (randomly selected) vectors in X.']);
    end
end

return
