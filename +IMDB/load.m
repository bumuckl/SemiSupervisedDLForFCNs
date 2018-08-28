function imdb = load(path)
%INIT load and prepare an IMDB
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

    load(path);
    %if round(max(imdb.images.data(:))) > 1
    %    imdb.images.data = normalize2(imdb.images.data, 'single', 8);
    %else
        imdb.images.data = single(imdb.images.data);
    %end
    imdb.images.labels = single(imdb.images.labels);
    
    % Add filename to meta if not available
    if ~isfield(imdb.meta, 'name') || ~isfield(imdb.meta, 'pathstr')
        [pathstr,name,ext] = fileparts(path);
        imdb.meta.name = name;
        imdb.meta.pathstr = pathstr;
    end
end

