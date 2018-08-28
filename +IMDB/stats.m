function stats( imdb )
%STATS 
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

		sz = size(imdb.images.data)
		
		disp(['Number of Images: ' num2str(sz(4))]);
		disp(['HxWxC: ' num2str(sz(1)) 'x' num2str(sz(2)) 'x' num2str(sz(3))]);
		disp(['Train-Val-Test Split: ' num2str(sum(imdb.images.set == 1)) '-' num2str(sum(imdb.images.set == 2)) '-' num2str(sum(imdb.images.set == 3)) '(' num2str(sum(imdb.images.set == 1)/sz(4)) '-' num2str(sum(imdb.images.set == 2)/sz(4)) '-' num2str(sum(imdb.images.set == 3)/sz(4)) ')']);
	
		% Class Statistics
        classes = unique(imdb.images.labels);
		num_classes = numel(classes);
		class_distributions = [];
		for c=1:num_classes
			class_distributions(c) = sum(imdb.images.labels(:) == c);
		end
		disp(['Number of Classes: ' num2str(num_classes) '(' sprintf('-%d',unique(imdb.images.labels)) ')']);
		disp(['Class Distribution: ' num2str(class_distributions) '(' num2str(class_distributions./(sum(class_distributions))) ')']);
	
		% Report per channel statistics
		for c=1:sz(3)
			tmp = imdb.images.data(:,:,c,:);
			disp(['Channel ' num2str(c) ' Stats: Min: ' num2str(min(tmp(:))) ' Mean: ' num2str(mean(tmp(:))) ' Median: ' num2str(median(tmp(:))) ' Std: ' num2str(std(tmp(:))) ' Max: ' num2str(max(tmp(:)))]);
			figure, imhist(tmp(:)), title(['Histogram for Channel ' num2str(c)]);
            
            % Report per label statistics
            for l=1:num_classes
                lidx = find(imdb.images.labels == classes(l));
                ltmp = tmp(lidx);
                disp(['Channel ' num2str(c) ', Label ' num2str(classes(l)) ' Stats: Min: ' num2str(min(ltmp(:))) ' Mean: ' num2str(mean(ltmp(:))) ' Median: ' num2str(median(ltmp(:))) ' Std: ' num2str(std(ltmp(:))) ' Max: ' num2str(max(ltmp(:)))]);
                figure, imhist(ltmp), title(['Histogram for Channel ' num2str(c), ', Label ' num2str(classes(l))]);
            end
        end
    
end

