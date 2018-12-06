function [portTimes, mapTimes] = getportfreq(filePath, numberOfPorts, ports, granularity, portTimes, mapTimes)
% getportfreq return the port frequence by seconds or minutes, according to
% the specified granularity
%
% SYNOPSIS: 
%
% INPUT filePath: Example:'\tmp\data.txt'
%       numberOfPorts: number of lines or ports of the file
%       ports: a vector of ports to be evealuated
%       granularity: granularity may be 'minutes' or 'seconds' only
%       portTimes: matrix nxm to store counted values and return as result
%       mapTimes: containers.Map to save original time and the new one
%
% OUTPUT [portTimes, mapTimes]: 
%
% EXAMPLE moseigensimgettime('\tmp\a.txt', 10, ports, granularity,
%                            portTimes, mapTimes)
%
% SEE ALSO 
%
% created with MATLAB R2016a on Ubuntu 16.04
% created by: Thiago Vieira
% DATE: 01-Oct-2010
%

for port = 1:numberOfPorts        
    fileID = fopen([filePath num2str(ports(port)) '.txt']);
    fileInfo = textscan(fileID,'%s %u');
    fclose(fileID);
    fileTimes = fileInfo{1};
    if size(fileTimes) > 0
        times = cell2mat(fileTimes);
        timeValues = fileInfo{2};
        if strcmp(granularity,'minutes')
             for time = 1:size(times,1)
                timeStr = strsplit(times(time,:),':');
                hh = str2num(timeStr{1});
                mm = str2num(timeStr{2});
                minute = (hh - 21) * 60 + mm + 1;   % a refactoring is requiredo to work with all cases
                portTimes(port,minute) = timeValues(time);
                if isKey(mapTimes,num2str(minute)) > 0
                    old = values(mapTimes,{num2str(minute)});
                    if old{1}{1} ~= timeStr{1} | old{1}{2} ~= timeStr{2}
                        warning(['minute=' num2str(minute) ', old=' old{1}{1} ', new=' timeStr{1}]);
                    end
                end
                mapTimes(num2str(minute)) = timeStr;
            end
        elseif strcmp(granularity,'seconds')
            for time = 1:size(times,1)
                timeStr = strsplit(times(time,:),':');
                hh = str2num(timeStr{1});
                mm = str2num(timeStr{2});
                ss = str2num(timeStr{3});
                second = ((hh - 21) * 60 * 60) + (mm * 60) + ss;
                portTimes(port,second) = timeValues(time);
                if isKey(mapTimes,num2str(second)) > 0
                    old = values(mapTimes,{num2str(second)});
                    if old{1}{1} ~= timeStr{1} | old{1}{2} ~= timeStr{2}
                        warning(['second=' num2str(second) ', old=' old{1}{1} ', new=' timeStr{1}]);
                    end
                end
                mapTimes(num2str(second)) = timeStr;
            end
        end           
    end
end