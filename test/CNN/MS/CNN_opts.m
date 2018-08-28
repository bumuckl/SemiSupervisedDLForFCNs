% Options file for this particular CNN
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

% Run Setup First
run ../../../Setup.m;

format long;

% Init the struct
options = struct;

% More specific options
options.debug = true;
options.verbose = false;
                          
% Other
options.data.dir = [MSSEGDATAPATH '/training'];
options.data.dirMSKRI = [MSKRIDATAPATH '/training'];
options.data.axis = 3; % 3 means Axial
options.data.patchsize = [48 48];
options.data.partitions = [0.7 0.3 0]; % Train, Val and Test Partitioning
options.data.intensityScale = 1; %:0.01:1.1;
options.data.noise_sigmas = 0;
options.data.mirror = 0;
options.data.rotations = 0; %:90:270;
options.data.scales = 1;
options.data.meanImage = false;
options.data.lowMemory = false;
options.data.randomCrops = true;
options.data.randomCropSize = [128 128];
options.data.randomCropsPerAugmentation = 6; % 12 for B,D, 6 for C
options.data.wholeSlices = false;
options.data.patchesPerAugmentation = 20;
options.data.domains = {'A'};
options.data.patients = {}; % 10-12 for MSKRI, 1-3 for Domain A, 4-6 for Domain B and so on

domainStr = '';
patientString = '';
if numel(options.data.domains) > 0
    domainStr = strjoin(options.data.domains, '');
else
    domainStr = 'All';
end
if numel(options.data.patients) > 0
    patientStr = ['_patients' strjoin(options.data.patients, '-')];
else
    patientStr = '';
end
if options.data.wholeSlices
    options.data.imdbDir = [IMDBPATH 'imdb_msseg_domain' domainStr patientStr ...
                         '_wholeslices' ...
                         '_r' sprintf('-%d',options.data.rotations) ...
						 '_is' sprintf('-%d',options.data.intensityScale) ...
                         '_s' sprintf('-%d',options.data.scales) ...
                         '_n' sprintf('-%d',options.data.noise_sigmas) ...
                         '.mat'];
elseif options.data.randomCrops
    options.data.imdbDir = [IMDBPATH 'imdb_msseg_domain' domainStr patientStr ...
                         '_c' num2str(options.data.randomCropSize(1)) 'x' num2str(options.data.randomCropSize(2)) ...
                         '_r' sprintf('-%d',options.data.rotations) ...
						 '_is' sprintf('-%d',options.data.intensityScale) ...
                         '_s' sprintf('-%d',options.data.scales) ...
                         '_n' sprintf('-%d',options.data.noise_sigmas) ...
                         '.mat'];
else
    options.data.imdbDir = [IMDBPATH 'imdb_msseg_domain' domainStr patientStr ...
                         '_p' num2str(options.data.patchsize(1)) 'x' num2str(options.data.patchsize(2)) ...
                         '_r' sprintf('-%d',options.data.rotations) ...
						 '_is' sprintf('-%d',options.data.intensityScale) ...
                         '_s' sprintf('-%d',options.data.scales) ...
                         '_n' sprintf('-%d',options.data.noise_sigmas) ...
                         '.mat'];
end
% Classes
options.data.classes.background = 1;
options.data.classes.lesion = 2;

% Net
options.net.lastLayerBeforeLoss = 'conv_u0d_score';

% Train
options.train.batchsize = 6;
options.train.classContributions = [0.7 0.3];
options.train.learningRate = 0.000001; %[0.01 0.01 0.01 0.01 0.01 0.001 0.001 0.001 0.001 0.001 0.0001 0.0001];
options.train.weightDecay = 0.0005;
options.train.momentum = 0.9;
options.train.numEpochs = 50;
options.train.embeddingLayers = {'conv_u0cd'}; % Or conv_d1bc (a layer before the dropout layer int he middle of U-Net
options.train.embeddingMargin = 1000;
options.train.maxNumEmbeddingsPerImage = 200;
options.train.derOutputs = {'objective', 1}; %, 'embedding_objective1', 1, 'embedding_objective2', 1};
options.train.gpus = [1];
options.train.solver = 'adagrad'; % Not implemented atm
options.train.expDir_prefix = './models';
options.train.netDescription = ['MSSEG_UNET_FscoreLoss_Domain' strjoin(options.data.domains, '')];
if options.data.wholeSlices
    options.train.expDir = [options.train.expDir_prefix ...
                        options.train.netDescription ...
                        '_wholeslices' ...
                        '_l' num2str(options.train.learningRate) ...
                        '_b' num2str(options.train.batchsize) ...
                        '/'];
elseif options.data.randomCrops
    options.train.expDir = [options.train.expDir_prefix ...
                        options.train.netDescription ...
                        '_c' num2str(options.data.randomCropSize(1)) 'x' num2str(options.data.randomCropSize(2)) ...
                        '_l' num2str(options.train.learningRate) ...
                        '_b' num2str(options.train.batchsize) ...
                        '/'];
else
    options.train.expDir = [options.train.expDir_prefix ...
                        options.train.netDescription ...
                        '_p' num2str(options.data.patchsize(1)) 'x' num2str(options.data.patchsize(2)) ...
                        '_l' num2str(options.train.learningRate) ...
                        '_b' num2str(options.train.batchsize) ...
                        '/'];
end

% Test
options.test.data.dir = [MSSEGDATAPATH '/testing'];
options.test.data.dirMSKRI = [MSKRIDATAPATH '/testing'];
options.test.threshold = 0.353;
options.test.embeddingLayers = options.train.embeddingLayers;
options.test.epoch = 'last';
options.test.tsne = true;
options.test.embeddingDomains = {'A', 'B'};
options.test.batchsize = 100;
options.test.domains = {};
options.test.patients = {};