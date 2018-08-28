function imdb = resize( imdb, size_or_scale )
%REPARTITION Batch resize all the images inside the provided imdbstruct
%using either a vector of target dimensions or a scale.
%
% INPUT:
%
%   imdb = the imdb struct that you want to repartition
%   size_or_scale = a 2 element vector of target dimensions or a scalar for
%   resizing
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

    % Gain access to all methods of the package
    import IMDB.*
    
    data = imdb.images.data;
    imdb.images.data =  zeros(0,0,0,0);
    for i=1:size(data,4)
        imdb.images.data(:,:,:,i) = imresize(data(:,:,:,i), size_or_scale);
    end
end

