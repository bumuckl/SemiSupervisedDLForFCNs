% Finetune an Existing Network
% Author: Christoph Baur

%clear;
close all;
run ../../../Setup.m

% Options
CNN_opts;

% Override options
options.train.fineTuneEpoch = '35';
options.train.fineTuneBaseline = [options.train.expDir_prefix '/MSBaseline/MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6/'];
options.train.fineTuneIMDB = [IMDBPATH '/' 'imdb_msseg_domainA_c128x128_r-0_is-1_s-1_n-0.mat'];
options.train.fineTuneDomain = 'A';
options.debug = true;
options.test.tsne = false;
options.test.pca = false;
options.test.pca3D = false;
options.test.savefig = true;
options.test.embeddings = false;
options.test.savePredictedVolumes = false;
options.test.savePredictedBinaryVolumes = false;
options.test.plotROC = false;
options.test.domains = {};
options.test.patients = {};
options.test.plotROC = false;

% Load the IMDB
imdb = IMDB.load(options.train.fineTuneIMDB);

% Finetune the net
    
% Load network architecture - lazy resets
% Load trained model
if strcmp('last', options.train.fineTuneEpoch)
    options.train.fineTuneEpoch = num2str(CNN.findLastCheckpoint(options.train.fineTuneBaseline));
    epochFile = [options.train.fineTuneBaseline 'net-epoch-' options.train.fineTuneEpoch '.mat'];
else
    epochFile = [options.train.fineTuneBaseline 'net-epoch-' options.train.fineTuneEpoch '.mat'];
end
load(epochFile, 'net', 'stats') ;
net = dagnn.DagNN.loadobj(net) ;
options.train.derOutputs = {'objective', 1}; % just another reset

% Generate options.train.expDir
options.train.expDir = [options.train.expDir_prefix ...
                    options.train.netDescription ...
                    '_c' num2str(options.data.randomCropSize(1)) 'x' num2str(options.data.randomCropSize(2)) ...
                    '_l' num2str(options.train.learningRate) ...
                    '_b' num2str(options.train.batchsize) ...
                    '_finetune' options.train.fineTuneDomain '_smallscale/'];

% Save the patched net and state to the expDir
net_ = net;
net = net_.saveobj() ;
if ~exist(options.train.expDir, 'dir')
    mkdir(options.train.expDir);
end
save([options.train.expDir 'net-epoch-' num2str(options.train.fineTuneEpoch) '.mat'], 'net', 'stats') ;

% Train the model with the auxiliary embeddings
[net,stats] = cnn_train_dag_fcn(net, imdb, @fn_getBatchDAGNN, ...
                    'expDir', options.train.expDir, ...
                    'batchsize', options.train.batchsize, ...
                    'learningRate', options.train.learningRate , ...
                    'numEpochs', options.train.numEpochs, ...
                    'weightDecay', options.train.weightDecay, ...
                    'gpus', options.train.gpus, ... 
                    'derOutputs', options.train.derOutputs, ...
                    'continue', true, ...
                    'plotStatistics', true ...
      );
%if ~exist([options.train.expDir '/eval-epoch-' num2str(options.train.numEpochs)], 'dir')
    [ Eval, embeddings ] = fn_test(options);
%end
