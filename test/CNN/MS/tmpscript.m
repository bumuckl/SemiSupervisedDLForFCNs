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

% Options
CNN_opts;

% Override options
% options.train.embeddingLoss = 'hadsell';
% options.train.embeddingLambda = [0.001];
% options.train.embeddingMargin = 1000;
options.train.numEmbeddings = [500]; % Note: 0 means "take all embeddings"
% options.train.embeddingLoss = 'cosine';
% options.train.embeddingLambda = [1]; % Usually, the derivatives are 1/100 of the relu derivatives right before the respective layer
% options.train.embeddingMargin = [1];
options.train.embeddingLoss = 'cosine';
options.train.embeddingLambda = [1]; % Usually, the derivatives are 1/100 of the relu derivatives right before the respective layer
options.train.embeddingMargin = [1];
options.train.samplingStrategy = {'fixed'};
options.train.samplingPartitions = {[0.8 0.2]};
options.train.lu_ratio = 1; % Ratio of unlabeled to labeled data, e.g. 2 means twice the unlabeled data
options.train.fineTuneEpoch = '35';
options.train.fineTuneBaseline = [options.train.expDir_prefix '/MSBaseline/MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6/'];
options.train.fineTuneIMDBl = [IMDBPATH '/' 'imdb_msseg_domainA_c128x128_r-0_is-1_s-1_n-0.mat'];
options.train.fineTuneIMDBu = [IMDBPATH '/' 'imdb_msseg_domainB_c128x128_r-0_is-1_s-1_n-0.mat'];
options.debug = true;
options.test.epoch = 'last'; % or 'last'
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
options.train.expDir_prefix = [options.train.expDir_prefix 'MSProofOfConcept\'];
options.train.numEpochs = 50;

% Load the IMDBs for labeled and unlabeled data
imdbl = IMDB.load(options.train.fineTuneIMDBl);
imdbu = IMDB.load(options.train.fineTuneIMDBu);

% Construct the actual IMDB used for the experiment
imdb = imdbl;
imdb.images.data_unlabeled = imdbu.images.data;
imdb.images.labels_unlabeled = imdbu.images.labels;
imdb.images.lu_ratio = options.train.lu_ratio;

% Train the nets
Evals = {};
for k=1:1
for s=1:length(options.train.samplingStrategy)   
for l=1:length(options.train.embeddingLambda)
	for n=1:length(options.train.numEmbeddings)
    for m=1:length(options.train.embeddingMargin)
    % All supervised embeddings together
    glob.kl_all = [];
    glob.js_all = [];
    
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
        if strcmp(options.train.embeddingLoss, 'CORAL')
           layername = ['coral_' options.train.embeddingLayers{e}];
           outputname = ['coral_objective' num2str(e)];
           net.addLayer('label_ignoreBG', dagnn.IgnoreBackgroundPixels(), {'input', 'label'}, {'label_ignoreBG'}, {});
           net.addLayer('label_lu_ignoreBG', dagnn.IgnoreBackgroundPixels(), {'input', 'label_lu'}, {'label_lu_ignoreBG'}, {});
           net.addLayer([options.train.embeddingLayers{e} '_resizelabels'], dagnn.ResizeLabels(), {options.train.embeddingLayers{e}, 'label_ignoreBG'}, {[options.train.embeddingLayers{e} '_resizedlabels']}, {});
           net.addLayer('embedding_sampler', dagnn.EmbeddingSampler('numEmbeddings', options.train.numEmbeddings(n), 'lu_ratio', options.train.lu_ratio, 'strategy', options.train.samplingStrategy{s}, 'partitions', options.train.samplingPartitions{s}), {options.train.embeddingLayers{e}, [options.train.embeddingLayers{e} '_resizedlabels'], 'label_lu_ignoreBG'}, {'embeddings', 'graph', 'embeddings_labels'}, {});
           net.addLayer(layername, dagnn.CORALLoss('lambda', options.train.embeddingLambda(l), 'numEmbeddings', options.train.numEmbeddings(n)), {'embeddings', 'embeddings_labels', 'graph'}, {outputname}, {});
        else
            layername = ['embedding_' options.train.embeddingLayers{e}];
            outputname = ['embedding_objective' num2str(e)];
            net.addLayer('label_ignoreBG', dagnn.IgnoreBackgroundPixels(), {'input', 'label'}, {'label_ignoreBG'}, {});
            net.addLayer('label_lu_ignoreBG', dagnn.IgnoreBackgroundPixels(), {'input', 'label_lu'}, {'label_lu_ignoreBG'}, {});
            net.addLayer([options.train.embeddingLayers{e} '_resizelabels'], dagnn.ResizeLabels(), {options.train.embeddingLayers{e}, 'label_ignoreBG'}, {[options.train.embeddingLayers{e} '_resizedlabels']}, {});
            net.addLayer('embedding_sampler', dagnn.EmbeddingSampler('numEmbeddings', options.train.numEmbeddings(n), 'lu_ratio', options.train.lu_ratio, 'strategy', options.train.samplingStrategy{s}, 'partitions', options.train.samplingPartitions{s}), {options.train.embeddingLayers{e}, [options.train.embeddingLayers{e} '_resizedlabels'], 'label_lu_ignoreBG'}, {'embeddings', 'graph'}, {});
            net.addLayer(layername, dagnn.SemiSupervisedEmbeddingLoss('lambda', options.train.embeddingLambda(l), 'margin', options.train.embeddingMargin(m), 'numEmbeddings', options.train.numEmbeddings(n), 'loss', options.train.embeddingLoss), {'embeddings', [options.train.embeddingLayers{e} '_resizedlabels'], 'graph'}, {outputname}, {});
        end
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
    modelname = [options.train.netDescription ...
                        '_c' num2str(options.data.randomCropSize(1)) 'x' num2str(options.data.randomCropSize(2)) ...
                        '_l' num2str(options.train.learningRate) ...
                        '_b' num2str(options.train.batchsize) ...
                        '_lambda' num2str(options.train.embeddingLambda(l)) ...
                        '_m' num2str(options.train.embeddingMargin(m)) ...
                        '_' strjoin(options.train.embeddingLayers,'_') ... 
                        '_' options.train.embeddingLoss ...
                        '_lu' num2str(options.train.lu_ratio) ...
                        '_finetuneAB_' options.train.samplingStrategy{s} vec2str(options.train.samplingPartitions{s}) '_full2small' num2str(options.train.numEmbeddings(n)) '_' num2str(k) '/'];
    options.train.expDir = [options.train.expDir_prefix ...
                        modelname '/'];
                    
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
		
	% Save KL stats
    disp(['KL-Stats: Mean: ' num2str(mean(glob.kl_all)) ' Stddev: ' num2str(std(glob.kl_all))]);
    disp(['JS-Stats: Mean: ' num2str(mean(glob.js_all)) ' Stddev: ' num2str(std(glob.js_all))]);
      
    if ~exist([options.train.expDir '/eval-epoch-' num2str(options.train.numEpochs)], 'dir')
          mkdir([options.train.expDir '/eval-epoch-' num2str(options.train.numEpochs)]);
    end
    if ~exist([options.train.expDir '/eval-epoch-' num2str(options.train.numEpochs) '/kl.mat'], 'file')
        kl_stats = struct;
        kl_stats.all = glob.kl_all;
        kl_stats.mean = mean(glob.kl_all);
        kl_stats.std = std(glob.kl_all);
        save([options.train.expDir '/eval-epoch-' num2str(options.train.numEpochs) '/kl.mat'], 'kl_stats');

        % Save box plot
        figure, boxplot(kl_stats.all);
        savefig([options.train.expDir '/eval-epoch-' num2str(options.train.numEpochs) '/kl.fig']);
        saveas(gcf, [options.train.expDir '/eval-epoch-' num2str(options.train.numEpochs) '/kl.png']);
        
        js_stats = struct;
        js_stats.all = glob.js_all;
        js_stats.mean = mean(glob.js_all);
        js_stats.std = std(glob.js_all);
        save([options.train.expDir '/eval-epoch-' num2str(options.train.numEpochs) '/js.mat'], 'js_stats');
        
        % Save box plot
        figure, boxplot(js_stats.all);
        savefig([options.train.expDir '/eval-epoch-' num2str(options.train.numEpochs) '/js.fig']);
        saveas(gcf, [options.train.expDir '/eval-epoch-' num2str(options.train.numEpochs) '/js.png']);
    end
			
    if ~exist([options.train.expDir '/eval-epoch-' num2str(options.train.numEpochs) '/Eval.mat'], 'file')
        [ Eval, embeddings ] = fn_test(options);
    else
        load([options.train.expDir '/eval-epoch-' num2str(options.train.numEpochs) '/Eval.mat']);
    end
    Eval.modelname = modelname;
    Evals{end+1} = Eval;
    
    end
    end
end
end
end

% Export all Evals to one Excel file
ExcelCell = {'Patient',	'Domain', 'TP', 'TN', 'FP', 'FN', 'Precision', 'Recall', 'Fscore', 'Error',	'Notes'};
for i=1:length(Evals)
    Eval = Evals{i};
    ExcelCell{end+1,1} = Eval.modelname;
    currentRow = size(ExcelCell,1);
    for row=1:size(Eval.table,2)
        ExcelCell(currentRow + row,:) = {Eval.table(row).patient, Eval.table(row).domain, Eval.table(row).TP, Eval.table(row).TN, Eval.table(row).FP, Eval.table(row).FN, Eval.table(row).Precision, Eval.table(row).Recall, Eval.table(row).Fscore, Eval.table(row).error, ''};
    end
end
xlswrite([options.train.expDir_prefix '/Report_' options.train.embeddingLoss '.xlsx'], ExcelCell);

clear;