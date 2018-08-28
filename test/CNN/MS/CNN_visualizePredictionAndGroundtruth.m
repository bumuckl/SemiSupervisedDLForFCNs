%Visualize a Prediction and Groundtruth next to each other
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

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