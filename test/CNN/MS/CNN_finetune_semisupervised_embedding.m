% Do some kind of grid search for models based on EmbeddingLoss
% (Supervised!)
% Comments marked with #1 mean uncomment this to embed 10-class numbers (this method uses EmbeddingLoss)
% Comments marked with #2 mean uncomment this to embed prime numbers (this method uses SemiSupervisedEmbeddingLoss without unlabeled data)
%
% Author: Christoph Baur

%clear;
close all;
run ../../../Setup.m

% Globals
global glob;
glob.kl_all = [];

% Options
CNN_opts;

% Override options
% options.train.embeddingLoss = 'hadsell';
% options.train.embeddingLambda = [0.001];
% options.train.embeddingMargin = 1000;
options.train.numEmbeddings = 100; % Note: 0 means "take all embeddings"
options.train.embeddingLoss = 'ACD';
options.train.embeddingLambda = [100]; % Usually, the derivatives are 1/100 of the relu derivatives right before the respective layer
options.train.embeddingMargin = [0.5];
options.train.samplingStrategy = 'fixed';
options.train.samplingPartitions = [0.8 0.2];
options.train.lu_ratio = 1; % Ratio of unlabeled to labeled data, e.g. 2 means twice the unlabeled data
options.train.fineTuneEpoch = '35';
options.train.fineTuneBaseline = [options.train.expDir_prefix '/MSBaseline/MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6/'];
options.train.fineTuneIMDBl = [IMDBPATH '/' 'imdb_msseg_domainA_c128x128_r-0_is-1_s-1_n-0.mat'];
options.train.fineTuneIMDBu = [IMDBPATH '/' 'imdb_msseg_domainD_patients11-12_c128x128_r-0_is-1_s-1_n-0-prior.mat'];
options.debug = true;
options.test.epoch = 'last'; % or 'last'
options.test.tsne = false;
options.test.pca = true;
options.test.pca3D = false;
options.test.savefig = true;
options.test.embeddings = false;
options.test.savePredictedVolumes = false;
options.test.savePredictedBinaryVolumes = false;
options.test.plotROC = false;
options.train.expDir_prefix = [options.train.expDir_prefix 'MSSemiSupervisedNCC\'];

% Load the IMDBs for labeled and unlabeled data
imdbl = IMDB.load(options.train.fineTuneIMDBl);
imdbu = IMDB.load(options.train.fineTuneIMDBu);

% Construct the actual IMDB used for the experiment
imdb = imdbl;
imdb.images.data_unlabeled = imdbu.images.data;
imdb.images.labels_unlabeled = imdbu.images.labels;
imdb.images.lu_ratio = options.train.lu_ratio;

% Train the nets
    
for l=1:length(options.train.embeddingLambda)
    for m=1:length(options.train.embeddingMargin)
    % All supervised embeddings together
    
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

    for e=1:length(options.train.embeddingLayers)
        layername = ['embedding_' options.train.embeddingLayers{e}];
        outputname = ['embedding_objective' num2str(e)];
        net.addLayer('label_ignoreBG', dagnn.IgnoreBackgroundPixels(), {'input', 'label'}, {'label_ignoreBG'}, {});
        net.addLayer('label_lu_ignoreBG', dagnn.IgnoreBackgroundPixels(), {'input', 'label_lu'}, {'label_lu_ignoreBG'}, {});
        net.addLayer([options.train.embeddingLayers{e} '_resizelabels'], dagnn.ResizeLabels(), {options.train.embeddingLayers{e}, 'label_ignoreBG'}, {[options.train.embeddingLayers{e} '_resizedlabels']}, {});
        net.addLayer('embedding_sampler', dagnn.EmbeddingSampler('numEmbeddings', options.train.numEmbeddings, 'lu_ratio', options.train.lu_ratio, 'strategy', options.train.samplingStrategy, 'partitions', options.train.samplingPartitions), {options.train.embeddingLayers{e}, [options.train.embeddingLayers{e} '_resizedlabels'], 'label_lu_ignoreBG'}, {'embeddings', 'graph'}, {});
        net.addLayer(layername, dagnn.SemiSupervisedEmbeddingLoss('lambda', options.train.embeddingLambda(l), 'margin', options.train.embeddingMargin(m), 'numEmbeddings', options.train.numEmbeddings, 'loss', options.train.embeddingLoss), {'embeddings', [options.train.embeddingLayers{e} '_resizedlabels'], 'graph'}, {outputname}, {});
        options.train.derOutputs{end+1} = outputname;
        options.train.derOutputs{end+1} = 1;
        if ~isfield(stats.train(1), outputname)
            for i=1:numel(stats.train)
                stats.train(i).(outputname) = 0;
            end
        end
        if ~isfield(stats.val(1), outputname)
            for i=1:numel(stats.train)
                stats.val(i).(outputname) = 0;
            end
        end
    end
    net.rebuild();

    % Generate options.train.expDir
    options.train.expDir = [options.train.expDir_prefix ...
                        options.train.netDescription ...
                        '_c' num2str(options.data.randomCropSize(1)) 'x' num2str(options.data.randomCropSize(2)) ...
                        '_l' num2str(options.train.learningRate) ...
                        '_b' num2str(options.train.batchsize) ...
                        '_lambda' num2str(options.train.embeddingLambda(l)) ...
                        '_m' num2str(options.train.embeddingMargin(m)) ...
                        '_' strjoin(options.train.embeddingLayers,'_') ... 
                        '_' options.train.embeddingLoss ...
                        '_lu' num2str(options.train.lu_ratio) ...
                        '_SSembeddingLoss' ...
                        '_finetuneAD_NCC_patient11-12/'];
                    
    % Save the patched net and state to the expDir
    net_ = net;
    net = net_.saveobj() ;
    if ~exist(options.train.expDir, 'dir')
        mkdir(options.train.expDir);
    end
    save([options.train.expDir 'net-epoch-' num2str(options.train.fineTuneEpoch) '.mat'], 'net', 'stats') ;

    % Train the model with the auxiliary embeddings
    [net,stats] = cnn_train_dag_fcn(net, imdb, @fn_getBatchDAGNNSemiSupervised, ...
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
      global glob;
      disp(['KL-Stats: Mean: ' num2str(mean(glob.kl_all)) ' Stddev: ' num2str(std(glob.kl_all))]);
    %if ~exist([options.train.expDir '/eval-epoch-' num2str(options.train.numEpochs)], 'dir')
        [ Eval, embeddings ] = fn_test(options);
    %end
    
    end
end

clear;