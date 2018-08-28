%FN_GETTESTEMBEDDINGIDX Given a test volume name, return a fixed number of indices for
%extracting embeddings. This should always be the same to ensure comparability. They are computed once, randomly.
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

function idx = fn_getTestEmbeddingIdx( volumeName, labelSlice, numEmbeddingsPerSlice, sliceNum )

fname = 'testEmbeddingIdx.mat';
testEmbeddingIdx = containers.Map;
if exist(fname, 'file')
    load(fname);
end

if testEmbeddingIdx.isKey(volumeName) && numel(testEmbeddingIdx(volumeName)) >= sliceNum
    vol = testEmbeddingIdx(volumeName);
    idx = vol{sliceNum};
else
    % Create the idx once
    intermediate_labels = labelSlice(:);
    classes = unique(intermediate_labels);
    labelsWithClass = [];
    for c=1:numel(classes)
        labelsWithClass(c) = sum(intermediate_labels == classes(c));
    end
    % Choose ridx according to numEmbeddingsPerSlice, keep the test ratio
    % balanced
    ridx = [];
    for c=1:numel(classes)
        embeddings_c = find(intermediate_labels == classes(c));
        ridx = [ridx; embeddings_c(randperm(numel(embeddings_c), min((1/numel(classes))*numEmbeddingsPerSlice, min(labelsWithClass) )))];
    end
    
    if testEmbeddingIdx.isKey(volumeName)
        vol = testEmbeddingIdx(volumeName);
    else
        vol = {};
    end
    vol{sliceNum} = ridx;
    testEmbeddingIdx(volumeName) = vol;
    save(fname, 'testEmbeddingIdx');
    idx = ridx;
end

end