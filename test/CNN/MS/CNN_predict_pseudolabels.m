% Creates a network architecture
%
% Author: Christoph Baur

clear;
close all;
run ../../../Setup.m

% Options
CNN_opts;

% Override option
options.train.fineTuneBaseline = [options.train.expDir_prefix '/MSBaseline/MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6/'];
options.debug = true;
options.data.domains = {'D'};
options.data.patients = {};
options.test.epoch = '35'; 
options.test.embeddings = false;
options.test.tsne = false;
options.test.pca = false;
options.test.pca3D = false;
options.test.savefig = false;
options.test.savePredictedVolumes = true;
options.test.saveSliceVisualizations = true;
options.test.savePredictedBinaryVolumes = false;
options.test.plotROC = false;
options.test.domains = {'D'};
options.test.patients = {};

% Predict on domain B,C and D training data to generate the pseudo label
% ground truth!
options.test.data.dir = options.data.dir;
options.test.data.dirMSKRI = options.data.dirMSKRI;
options.train.expDir = options.train.fineTuneBaseline;
[ Eval, embeddings ] = fn_test(options);