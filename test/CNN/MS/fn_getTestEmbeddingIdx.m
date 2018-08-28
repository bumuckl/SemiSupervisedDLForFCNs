function idx = fn_getTestEmbeddingIdx( volumeName, labelSlice, numEmbeddingsPerSlice, sliceNum )
%UNTITLED Given a test volume name, return a fixed number of indices for
%extracting embeddings. This should always be the same to ensure
%comparability. They are computed once, randomly.

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