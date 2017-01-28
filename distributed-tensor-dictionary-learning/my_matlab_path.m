function p = my_matlab_path(name)
% my_matlab_path   Return path on this computer for one of my catalogs
%
% use: p = my_matlab_path(name);     % where name may be as below
%                   m-files : common, comp, dlt, ict, tct
%                  packages : maxflow
%                  subjects : ele610, mik130
%              texture data : dictionaries, results, vectors, textures, test
% some special texture data : .d\aug12, .r\aug12, .d\jan13, .r\jan13, etc.
%                    images : BSDStest, BSDStrain, USC
%
% the BSD (Berkeley Segmentation Dataset) catalog should contain images from:
%   http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/
% the USC catalog should contain the images (both as tif and bmp) from: 
%   http://www.dip.ee.uct.ac.za/imageproc/stdimages/greyscale/
%
% ex: p = my_matlab_path('dlt');

% Karl Skretting, December 2012, January 2013

mfile = 'my_matlab_path';
if (nargin < 1)
    name = 'common';
end

if strcmpi(name,'dle')
    name = 'dlt';
end
    
if isunix
    p0 = '/home/thiago/dev/projects/anomaly-detection/distributed-tensor-dictionary-learning/';  % first part of path where most files should be
    if strcmpi(name,'dlt')
        p = '~/public_html/dle';
    elseif strcmpi(name,'ict')
        p = '~/public_html/ICTools';
    elseif (strcmpi(name,'common') || strcmpi(name,'comp') || strcmpi(name,'tct') )
        p = [p0,lower(name)];   
    elseif (strcmpi(name,'java') || strcmpi(name,'javaclasses'))
        p = [p0,'javaclasses'];   
    elseif strcmpi(name,'maxflow') 
        p = [p0,'maxflow-v3.01'];   
    elseif strcmpi(name,'spams') 
        p = [p0,'spams-matlab'];   
        
    else %  data file
        p = '~/matlab/tctdata/';   % where data is stored
        if (strcmpi(name,'tex') || strcmpi(name,'textures') || strcmpi(name,'test'))
            p = [p,'textures'];
        elseif (strcmpi(name,'dict') || strcmpi(name,'dictionaries'))
            p = [p,'dictionaries'];
        elseif (strcmpi(name,'res') || strcmpi(name,'results'))
            p = [p,'results'];
        elseif (strcmpi(name,'vec') || strcmpi(name,'vectors'))
            p = [p,'vectors'];
        elseif (strcmpi(name,'BSDStest') || strcmpi(name,'BSDSte'))
            p = '~/matlab/BSDS_images/test';   
        elseif (strcmpi(name,'BSDStrain') || strcmpi(name,'BSDStr'))
            p = '~/matlab/BSDS_images/train';   
        elseif strcmpi(name,'USC') 
            p = '~/matlab/USC_images';   
        elseif (name(1) == '.') || (name(1) == '/') || (name(1) == '\')      
            if (lower(name(2)) == 'd')
                p = [p,'dictionaries/',name(4:end)];
            elseif (lower(name(2)) == 'r')
                p = [p,'results/',name(4:end)];
            elseif (lower(name(2)) == 't')
                p = [p,'textures/',name(4:end)];
            elseif (lower(name(2)) == 'v')
                p = [p,'vectors/',name(4:end)];
            else
                p = [p,name(2:end)];
            end
        else
            p = [p,name];
        end
    end
else     % her for PC, Dropbox may be mounted different places
    p0 = which(mfile);
    i = strfind(p0,'\'); i = i(end-1);
    p0 = p0(1:i);  % only first part of path including '\' as last char
    if (strcmpi(name,'common') || strcmpi(name,'comp') || strcmpi(name,'dlt') || ...
        strcmpi(name,'ict') || strcmpi(name,'tct') )
        p = [p0,lower(name)];   
    elseif (strcmpi(name,'ele610') || strcmpi(name,'mik130'))
        p = [p0,lower(name)];   
    elseif (strcmpi(name,'java') || strcmpi(name,'javaclasses'))
        p = [p0,'javaclasses'];   
    elseif strcmpi(name,'maxflow') 
        p = [p0,'maxflow-v3.01'];   
    elseif strcmpi(name,'spams') 
        p = [p0,'spams-matlab'];           
    else %  data file (local if exist)
        if exist('C:\Users\Karl\Documents\MATLAB\tctdata','dir')
            p = 'C:\Users\Karl\Documents\MATLAB\tctdata\';
        elseif exist('C:\Users\Karl\Mine dokumenter\MATLAB\tctdata','dir')
            p = 'C:\Users\Karl\Mine dokumenter\MATLAB\tctdata\';
        elseif exist('D:\Karl\tctdata','dir')
            p = 'D:\Karl\tctdata\';
        else
            p = [p0,'tctdata\'];   % last try, perhaps on Dropbox folder ?
        end
        %
        if (strcmpi(name,'tex') || strcmpi(name,'textures') || strcmpi(name,'test'))
            p = [p,'textures'];
        elseif (strcmpi(name,'dict') || strcmpi(name,'dictionaries'))
            p = [p,'dictionaries'];
        elseif (strcmpi(name,'res') || strcmpi(name,'results'))
            p = [p,'results'];
        elseif (strcmpi(name,'vec') || strcmpi(name,'vectors'))
            p = [p,'vectors'];
        elseif (strcmpi(name,'BSDStest') || strcmpi(name,'BSDSte'))
            p = 'D:\Bilder\BSDS300\images\test';   
            if ~exist(p,'dir'); p = 'C:\Bilder\BSDS300\images\test'; end;
        elseif (strcmpi(name,'BSDStrain') || strcmpi(name,'BSDStr'))
            p = 'D:\Bilder\BSDS300\images\train';   
            if ~exist(p,'dir'); p = 'C:\Bilder\BSDS300\images\train'; end;
        elseif strcmpi(name,'USC')
            p = 'D:\Bilder\USC_tiff';   
            if ~exist(p,'dir'); p = 'C:\Bilder\USC_tiff'; end;
            if ~exist(p,'dir'); p = 'C:\Bilder\USC'; end;
            if ~exist(p,'dir'); p = 'D:\Bilder\USC'; end;
        elseif (name(1) == '.') || (name(1) == '/') || (name(1) == '\')      
            if (lower(name(2)) == 'd')
                p = [p,'dictionaries\',name(4:end)];
            elseif (lower(name(2)) == 'r')
                p = [p,'results\',name(4:end)];
            elseif (lower(name(2)) == 't')
                p = [p,'textures\',name(4:end)];
            elseif (lower(name(2)) == 'v')
                p = [p,'vectors\',name(4:end)];
            else
                p = [p,name(2:end)];
            end
        else
            p = [p,name];
        end
    end
end
    
if ~exist(p,'dir')
    disp([mfile,': ',p,' does not exist.']);
    disp(' -> just return pwd() ');
    p = pwd;
end

return    

