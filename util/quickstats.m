function stats = quickstats( tensor )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    stats = struct;
    stats.size = size(tensor);
    stats.min = gather(min(tensor(:)));
    stats.max = gather(max(tensor(:)));
    stats.median = gather(median(tensor(:)));
    stats.mean = gather(mean(tensor(:)));
    stats.unique = length(unique(tensor(:)));
    stats.norm = norm(tensor);
    
    string = '';
    string = [string 'Size: ' num2str(stats.size) char(10)];
    string = [string 'Min: ' num2str(stats.min) char(10)];
    string = [string 'Max: ' num2str(stats.max) char(10)];
    string = [string 'Mean: ' num2str(stats.mean) char(10)];
    string = [string 'Median: ' num2str(stats.median) char(10)];
    string = [string 'Unique: ' num2str(stats.unique) char(10)];
    string = [string 'Norm: ' num2str(stats.norm) char(10)];
    
    figure, imhist(tensor(:)), title('Histogram');
    
    disp(string);
end