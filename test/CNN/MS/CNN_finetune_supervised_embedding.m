% Do some kind of grid search for models based on EmbeddingLoss
% (Supervised!)
% Comments marked with #1 mean uncomment this to embed 10-class numbers (this method uses EmbeddingLoss)
% Comments marked with #2 mean uncomment this to embed prime numbers (this method uses SemiSupervisedEmbeddingLoss without unlabeled data)
%
% Author: Christoph Baur

%clear;
close all;
run ../../../Setup.m

% Options
CNN_opts;

% Override options
options.train.embeddingLoss = 'hadsell';
options.train.embeddingLambda = [0.01];
options.train.embeddingMargin = [1000];
% options.train.numEmbeddings = 200; % Note: 0 means "take all embeddings"
% options.train.embeddingLoss = 'cosine';
% options.train.embeddingLambda = [1];
% options.train.embeddingMargin = [1];
options.train.numEmbeddings = 100; % Note: 0 means "take all embeddings"
options.train.fineTuneEpoch = '35';
options.train.fineTuneBaseline = '../../../models/MSBaseline/MSSEG_UNET_FbetaLossWithUpdate_DomainC_c128x128_l1e-06_b6/';
options.train.fineTuneIMDB = [IMDBPATH '/' 'imdb_msseg_domainC_c128x128_r-0_is-1_s-1_n-0.mat'];
options.debug = true;
options.test.epoch = 'last'; % or 'last'
options.test.tsne = false;
options.test.pca = false;
options.test.savefig = true;
options.test.embeddings = false;
options.test.savePredictedVolumes = false;
options.test.savePredictedBinaryVolumes = false;
options.test.plotROC = false;

% Load the IMDB
imdb = IMDB.load(options.train.fineTuneIMDB);

% Train the nets
    
for l=1:length(options.train.embeddingLambda)
    for m=1:length(options.train.embeddingMargin)
    
    % B) Train a network with an embedding, but the prior is equal to the
    % actual labels and classes
    
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
        %net.addLayer('label_ignoreBG', dagnn.IgnoreBackgroundPixels(), {'input', 'label'}, {'label_ignoreBG'}, {});
        net.addLayer([options.train.embeddingLayers{e} '_resizelabels'], dagnn.ResizeLabels(), {options.train.embeddingLayers{e}, 'label_ignoreBG'}, {[options.train.embeddingLayers{e} '_resizedlabels']}, {});
        net.addLayer(layername, dagnn.EmbeddingLoss('lambda', options.train.embeddingLambda(l), 'margin', options.train.embeddingMargin(m), 'numEmbeddings', options.train.numEmbeddings, 'loss', options.train.embeddingLoss), {options.train.embeddingLayers{e}, [options.train.embeddingLayers{e} '_resizedlabels']}, {outputname}, {});
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
    options.train.expDir = [options.train.expDir_prefix 'MSSupervised/' ...
                        options.train.netDescription ...
                        '_c' num2str(options.data.randomCropSize(1)) 'x' num2str(options.data.randomCropSize(2)) ...
                        '_l' num2str(options.train.learningRate) ...
                        '_b' num2str(options.train.batchsize) ...
                        '_lambda' num2str(options.train.embeddingLambda(l)) ...
                        '_m' num2str(options.train.embeddingMargin(m)) ...
                        '_' strjoin(options.train.embeddingLayers,'_') ... 
                        '_' options.train.embeddingLoss ...
                        '_embeddingLoss' ...
                        '_finetuneC_80t020_ignoreBG/'];
                    
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
                        'learningRate', options.train.learningRate *0.01, ...
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
    
    end
end
