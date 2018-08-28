% Creates a network architecture
%
% Author: Christoph Baur

clear;
close all;
run ../../../Setup.m

% Options
CNN_opts;

% Override option
options.visu.predictedNII = [options.train.expDir_prefix '/MSSEG_UNET_FbetaLossWithUpdate_DomainAB_c128x128_l1e-06_b6/eval-epoch-50/01040VANE.nii'];
options.visu.groundtruthNII = [MSSEGDATAPATH '/testing/01040VANE/Consensus.nii.gz'];
options.visu.axis = 3;

% Do the test
predictedNII = NII(options.visu.predictedNII);
groundtruthNII = NII(options.visu.groundtruthNII);

predictedNII.compareToNII(groundtruthNII, options.visu.axis);