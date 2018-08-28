% Creates a network architecture
%
% Author: Christoph Baur

global BN;
BN.train = struct;
BN.train.mu = [];
BN.train.std = [];
BN.val = struct;
BN.val.mu = [];
BN.val.std = [];
BN.test = struct;
BN.test.mu = [];
BN.test.std = [];

clear;
close all;
run ../../../Setup.m

% Options
CNN_opts;
options.debug = true;
options.test.epoch = 'last'; %'10' or 'last'
options.test.embeddings = false;
options.test.tsne = false;
options.test.pca = false;
options.test.pca3D = false;
options.test.savefig = true;
options.test.plotROC = false;
options.test.savePredictedVolumes = false;
options.test.savePredictedBinaryVolumes = false;
options.train.numEpochs = 50;

% Load network architecture
CNN_network_UNET_MS_CB_Bnorm;

% Load the IMDB
% if ~exist('imdb', 'var')
%options.data.imdbDir = [IMDBPATH 'imdb_msseg_domainB_c128x128_r-0_is-1_s-1_n-0.mat'];
    imdb = IMDB.load(options.data.imdbDir);
% end
% lol = find(imdb.images.set == 1);
% imdb.images.set(lol(25:end)) = 2;

% Create the model folder and save the options file so we can actually
% determine the options behind each model
if ~exist(options.train.expDir, 'dir'), mkdir(options.train.expDir) ; end
save([options.train.expDir '/options.mat'], 'options');

% Train the net
[net,stats] = cnn_train_dag_fcn(net, imdb, @fn_getBatchDAGNN, ...
                            'expDir', options.train.expDir, ...
                            'batchsize', options.train.batchsize, ...
                            'learningRate', options.train.learningRate, ...
                            'numEpochs', options.train.numEpochs, ...
                            'weightDecay', options.train.weightDecay, ...
                            'gpus', options.train.gpus, ... 
							'derOutputs', options.train.derOutputs, ...
                            'continue', true, ...
                            'plotStatistics', true ...
              );
          
[ Eval, embeddings ] = fn_test(options);
