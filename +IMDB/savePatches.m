function savePatches( imdb, path )
%SAVEPATCHES
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

    if isfield(imdb.images,'filenames') && length(imdb.images.filenames) > 0
        numPatches = length(imdb.images.filenames);
        for i=1:numPatches
           file = imread([imdb.meta.name '/' imdb.images.filenames{i}]);
           imwrite(file, [path '/' num2str(i) '.png']);
        end
    else
        numPatches = size(imdb.images.data, 4);
        for i=1:numPatches
           file = squeeze(imdb.images.data(:,:,:,i));
           imwrite(file, [path '/' num2str(i) '.png']);
        end
    end
end

