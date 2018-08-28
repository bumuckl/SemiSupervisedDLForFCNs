function imdb = merge( imdb_a, imdb_b )
%MERGE Merge two imdb structs
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

    % Gain access to all methods of the package
    import IMDB.*
    
    % Make copy of imdb_a
    imdb = imdb_a;
    
    % Append imdb_b data
    imdb.images.data(:,:,:, end+1:end+size(imdb_b.images.data, 4)) = imdb_b.images.data;
    imdb.images.labels(1, end+1:end+size(imdb_b.images.labels, 2)) = imdb_b.images.labels;
    imdb.images.set(1, end+1:end+size(imdb_b.images.set, 2)) = imdb_b.images.set;
    
    % Set classes
    imdb.meta.classes = unique(imdb.images.labels);
    
    % Mean training data
    imdb.images.dataMean = mean(imdb.images.data(:,:,:,find(imdb.images.set == 1)), 4);
end

