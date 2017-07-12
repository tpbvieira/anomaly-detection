function res = ex31prop(dict, varargin)
% ex31prop        Display properties for dictionaries stored in ex31*.mat files.
%                 This is an example that display dictionary properties.
% This file may be adapted to display the properties you want to display,
% and in the format you find more appropriate.
% 
% Use:
%   res = ex31prop(dict);
%   res = ex31prop(dict, 'useLaTeX',true);
%-------------------------------------------------------------------------
% Arguments:
%   res         a cell arry with the results for each dictionary 
%   dict        name of mat-files, a cell array of names or a char array
%               or string (that may include wildchars)
%   There may be an additional number of input arguments, 
%   a struct, a cell or as pairs: argName, argVal, ...
%     'useLaTeX' true display the text-results in 'LaTeX' table format
%-------------------------------------------------------------------------
% Examples: 
%   res = ex31prop('ex311Aug081555.mat');
%   res = ex31prop('ex31*.mat');

%% ---------------------------------------------------------------------
% Copyright (c) 2009.  Karl Skretting.  All rights reserved.
% University of Stavanger.
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.his.no/~karlsk/
% 
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  30.10.2009  KS: function started
% Ver. 1.1  09.11.2009  KS: function extended
% Ver. 2.0  13.01.2010  KS: function simlified (old version is ex331.m)
% Ver. 2.1  27.04.2010  KS: also display multi-dictionary from ex314.m
% Ver. 2.2  09.08.2011  KS: simplified more, (old version is ex321.m)
% ---------------------------------------------------------------------
    
%% some special examples
if (nargin < 1)
    error('ex31prop: wrong number of arguments, see help.');
end

%% filenames should be char or cell
if ischar(dict)  % dict should be char or cell
    filenames = cell(size(dict,1), 1);
    for i = 1:size(dict,1) 
        filenames{i} = strtrim(dict(i,:));
    end
else 
    filenames = dict;
end
if ~iscell(filenames)
    res = 'ex31prop: dict (filenames) not in expected format, see help.';
    error(res);
end
% find number of files    
numfiles = 0;
for i = 1:numel(filenames)
    d = dir(filenames{i});
    numfiles = numfiles + numel(d);
end
usefiles = cell(numfiles,1);   
i = 0; j = 0;
while j < numfiles
    i = i+1;
    d = dir(filenames{i});
    for ii = 1:numel(d);
        j = j+1;
        usefiles{j} = d(ii).name;
        if (j >= numfiles); break; end;
    end
end
res = cell(numfiles, 1);

%% additional arguments/options 
uselatex = false;

%% get the additional options
nofOptions = nargin-1;
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
    if strcmpi(opName,'makeFigure') 
        if isnumeric(opVal) 
            makeFig = opVal; 
        elseif islogical(opVal) 
            makeFig = double(opVal); 
        else
            disp('ex31prop: Option makeFigure should be numeric.');
        end
    end
    if strcmpi(opName,'useLaTeX') 
        if isnumeric(opVal) 
            uselatex = (opVal ~= 0); 
        elseif islogical(opVal) 
            uselatex = opVal;
        else
            disp('ex31prop: Option useLaTeX should be logical (or numeric).');
        end
    end
end

%% make the results
for i=1:numfiles
    if (exist(usefiles{i}, 'file') == 2)
        d = dir(usefiles{i});
        Ds = load(usefiles{i});
        if strcmpi(Ds.transform,'ks2')  % an old name for my special m79 wavelet variant
            Ds.transform = 'm79';
        end
        if ~isfield(Ds, 'ResultFile')
            Ds.ResultFile = usefiles{i};
        end
        % Design method
        if strcmpi(Ds.ResultFile(1:min(5,length(Ds.ResultFile))),'ex311')
            method = 'ILS-DLA';
        elseif strcmpi(Ds.ResultFile(1:min(5,length(Ds.ResultFile))),'ex312')
            method = 'RLS-DLA';
        elseif strcmpi(Ds.ResultFile(1:min(5,length(Ds.ResultFile))),'ex313')
            method = 'K-SVD';
        elseif strcmpi(Ds.ResultFile(1:min(5,length(Ds.ResultFile))),'ex314')
            if ~isfield(Ds,'M'); Ds.M = size(Ds.D,3); end;
            method = [int2str(Ds.M),'-RLS-DLA'];
        elseif strcmpi(Ds.ResultFile(1:min(5,length(Ds.ResultFile))),'ex315')
            method = 'Sep.ILS-DLA';
        else
            method = 'unknown';
        end
        %
        if strcmpi(Ds.ResultFile(1:min(5,length(Ds.ResultFile))),'ex314')
            A = zeros(1,Ds.M); B = A;
            betamin = A; betaavg = A;
            for m=1:Ds.M
                propD = dictprop(Ds.D(:,:,m), false);
                A(m) = propD.A;
                B(m) = propD.B;
                betamin(m) = propD.betamin;
                betaavg(m) = propD.betaavg;
            end
            res{i} = struct('transform', Ds.transform, ...
                'K',Ds.K, ...
                'targetPSNR',Ds.targetPSNR, ...
                'N',Ds.N, ...
                'L',Ds.L, ...
                'ResultFile', Ds.ResultFile, ...
                'tabPSNR', Ds.tabPSNR, ...
                'tabNNZ', Ds.tabNNZ, ...
                'date', d.date(1:11), ...
                'method', method, ...
                'A', A, ...
                'B', B, ...
                'betamin', betamin, ...
                'betaavg', betaavg );
            if isfield(Ds,'tabIT')
                res{i}.tabIT = Ds.tabIT;
            end
        else
            propD = dictprop(Ds.D, false);
            res{i} = struct('transform', Ds.transform, ...
                'K',Ds.K, ...
                'targetPSNR',Ds.targetPSNR, ...
                'N',Ds.N, ...
                'L',Ds.L, ...
                'ResultFile', Ds.ResultFile, ...
                'tabPSNR', Ds.tabPSNR, ...
                'tabNNZ', Ds.tabNNZ, ...
                'date', d.date(1:11), ...
                'method', method, ...
                'prop', propD, ...
                'A', propD.A, ...
                'B', propD.B, ...
                'betamin', propD.betamin, ...
                'betaavg', propD.betaavg );
            if isfield(Ds,'tabIT')
                res{i}.tabIT = Ds.tabIT;
            end
        end
        %
    else
        res{i} = ['File ',usefiles{i},' does not exist.'];
    end
end

%% display som results as text, only res is needed
disp(' ');
disp(['ex31prop display properties for ',int2str(numfiles),' dictionary files.']);

if uselatex
    t = 'Dictionary filename   & iter & tPSNR & trans. &     A &     B & betamin & betaavg \\';
    I = find(t == '&');
    t0 = '\hline';
else
    t = 'Dictionary filename   |    method   | iter | tPSNR |       date  | trans. |    size |     A |     B | betamin | betaavg |';
    I = find(t == '|');
    t0 = char('-'*ones(1,numel(t))); t0(t == '|') = '+';
end
disp(t0);
disp(t);
disp(t0);
for i=1:numel(res)
    if uselatex
        t = blanks(numel(t));
        t((end-1):end) = '\\';
        t(I) = '&';
    else
        t = blanks(numel(t0));
        t(I) = '|';
    end
    j = 0;
    t(2:(numel(res{i}.ResultFile)+1)) = res{i}.ResultFile;
    if ~uselatex
        j = j+1; w = numel(res{i}.method);
        t((I(j)+2):(I(j)+w+1)) = res{i}.method;
    end
    j = j+1; w = 4;
    if isfield(res{i},'tabIT')
        t((I(j)+2):(I(j)+w+1)) = sprintf('%4i',res{i}.tabIT(end));
    else
        t((I(j)+2):(I(j)+w+1)) = sprintf('%4i',numel(res{i}.tabPSNR));
    end
    j = j+1; w = 5;
    t((I(j)+2):(I(j)+w+1)) = sprintf('%5.1f',res{i}.targetPSNR);
    if ~uselatex
        j = j+1; w = numel(res{i}.date);
        t((I(j)+2):(I(j)+w+1)) = res{i}.date;
    end
    j = j+1; w = numel(res{i}.transform);
    t((I(j)+2):(I(j)+w+1)) = res{i}.transform;
    if ~uselatex
        j = j+1; w = 7;
        t((I(j)+2):(I(j)+w+1)) = [sprintf('%3i',res{i}.N),'x',sprintf('%3i',res{i}.K)];
    end
    j0 = j+1;
    for m=1:numel(res{i}.A)
        j = j0; w = 5;
        t((I(j)+2):(I(j)+w+1)) = sprintf('%5.2f',res{i}.A(m));
        j = j+1; w = 5;
        t((I(j)+2):(I(j)+w+1)) = sprintf('%5.2f',res{i}.B(m));
        j = j+1; w = 7;
        t((I(j)+2):(I(j)+w+1)) = sprintf('%7.2f',res{i}.betamin(m));
        j = j+1; w = 7;
        t((I(j)+2):(I(j)+w+1)) = sprintf('%7.2f',res{i}.betaavg(m));
        disp(t);
    end
end
disp(t0);

return




