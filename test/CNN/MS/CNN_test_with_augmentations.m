% Creates a network architecture
%
% Author: Christoph Baur

clear;
close all;
run ../../../Setup.m

% Options
CNN_opts;

% Override option
options.debug = true;
options.test.evalFolderPrefix = 'eval_augmented';
options.test.epoch = 'last'; %'10' or 'last'
options.test.embeddings = false;
options.test.tsne = false;
options.test.pca = true;
options.test.savefig = true;
options.test.savePredictedVolumes = false;
options.test.plotROC = false;
options.train.expDir = [options.train.expDir_prefix '/MSBaseline/MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6/'];

% Set augmentations
options.test.augmentations = struct;
options.test.augmentations.mirror = 0;
options.test.augmentations.rotations = -60:15:60;
options.test.augmentations.scales = 1;

% Do the test
[ Eval, embeddings ] = fn_test(options);