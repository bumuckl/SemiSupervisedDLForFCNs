% Take the baseline network trained for domain A, feed it with data from all domains and visualize the embeddings to see if classes
% from different domains still land in the same cluster or not.
%
% Author: Christoph Baur

clear;
close all;
run ../../../Setup.m

% Options
CNN_opts;

% Override option
options.debug = true;
options.data.domains = {'A', 'B', 'C', 'D'};
options.data.patients = {};
options.test.epoch = 'last'; %'10' or 'last'
options.test.embeddings = true;
options.test.tsne = false;
options.test.embeddingDomains = {'A', 'B', 'C', 'D'};
options.test.pca = false;
options.test.pca3D = true;
options.test.savefig = true;
options.test.savePredictedVolumes = false;
options.test.saveSliceVisualizations = false;
options.test.savePredictedBinaryVolumes = false;
options.test.plotROC = false;
options.test.models = {
     [options.train.expDir_prefix '/MSBaseline/MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6/'],
};

% Do the test
for m=1:length(options.test.models)
    options.train.expDir = options.test.models{m};
    [ Eval, embeddings ] = fn_test(options);
end