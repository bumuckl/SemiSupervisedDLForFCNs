function imdb = init()
%INIT Returns a totally empty imdb struct. This is for convenience to make
%sure you know what this struct looks like
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

    % Gain access to all methods of the package
    import IMDB.*

    imdb = struct;
    imdb.images = struct;
    imdb.images.data = zeros(0,0,0,0);
    imdb.images.dataMean = [];
    imdb.images.labels = [];
    imdb.images.set = [];
    imdb.images.filenames = {};
    imdb.meta.sets = {'train', 'val', 'test'};
    imdb.meta.classes = [];
end

