% java_access      Check access to java-class (package) mpv2.DictionaryLearning 
% that is used by for example the function sparseapprox.m
% On my computer the directories are
%   Development:  D:\Karl\DIV07\java\mpv2  and jc.bat compile all classes
%   Compiled classes:  F:\matlab\javaclasses\
%   or somewhere close to current directory for Unix
% These commands should display nothing if ok.

% name of the important class
viktigKlasse = 'mpv2.DictionaryLearning';

% check access
% disp('-- Current version of Java is');
% version -java
if not(exist(viktigKlasse,'class'))   
    disp(['No access to ',viktigKlasse,' in static or dynamic Java path.']);
    disp('Set dynamic Java path (works only if supported by the Matlab-version).');
    javaclasspath( my_matlab_path('java') );
    
    if not(exist(viktigKlasse,'class'))       % try the old code
        disp('  -- OBS -- not correct in my_matlab_path() --- try more ... ');
        if isunix
            if exist('../javaclasses/','dir')
                disp('Set dynamic Java path to ../javaclasses/');
                javaclasspath('../javaclasses/');
            elseif exist('../../Matlab/javaclasses/','dir')
                disp('Set dynamic Java path to ../../Matlab/javaclasses/');
                javaclasspath('../../Matlab/javaclasses/');
            elseif exist('../../../Matlab/javaclasses/','dir')
                disp('Set dynamic Java path to ../../../Matlab/javaclasses/');
                javaclasspath('../../../Matlab/javaclasses/');
            end
        else
            if exist('F:/matlab/javaclasses/','dir')
                disp('Set dynamic Java path to F:/matlab/javaclasses/');
                javaclasspath('F:/matlab/javaclasses/');
            elseif exist('D:/Karl/Matlab/javaclasses/','dir')
                disp('Set dynamic Java path to D:/Karl/Matlab/javaclasses/');
                javaclasspath('D:/Karl/Matlab/javaclasses/');
            end
            % clear all; javarmpath('F:/matlab/javaclasses/');
            % javaaddpath('D:/Karl/Matlab/javaclasses/');
        end
    end
    %
    if not(exist(viktigKlasse,'class'))
        disp(['Still no access to ',viktigKlasse,' in static or dynamic Java path.']);
        return;
    else
        disp(['Java access OK, ',viktigKlasse,' is in static or dynamic Java path.']);
    end
end

return
