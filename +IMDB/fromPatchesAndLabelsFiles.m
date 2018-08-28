function imdb = fromPatchesAndLabelsFiles(path, set_partitions)
%FROMPATCHESANDLABELS Read all .mat files in path and convert the contained patches and labels variables returned from
%"CreatePatchesData" into an IMDB struct (i.e. used by MatConvNet)
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
    
    % Gain access to all methods of the package
    import IMDB.*

    if nargin < 2
        set_partitions = [0.6 0.2 0.2];
    end

    imdb = init();
    
    % Read all files, iterate over all files, obtain an imdb_ struct for
    % every image and merge it with the overall imdb struct
    files = getAllFiles(path, '*.mat', 1);
    for f=1:length(files)
        load(files{f});
        tmp_patches{1} = patches{f};
        tmp_labels{1} = labels{f};
        imdb_ = fromPatchesAndLabels(tmp_patches, tmp_labels, set_partitions);
        imdb = merge(imdb, imdb_);
        clearvars tmp_patches tmp_labels imdb_
        disp(['Done with file ' files{f} '(file ' num2str(f) ' of ' num2str(length(files)) ' files)']);
    end
end

