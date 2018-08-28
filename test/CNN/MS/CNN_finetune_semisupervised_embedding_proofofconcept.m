% >the script used to train the models for the Proof-of-Concept experiments reported in the paper. 
% We utilize the ground-truth-labels here to generate a perfect prior. Finetune an existing Network
% using both labeled and "unlabeled" data. Do it in a "grid search" way with different params to 
% get a comparison of different models
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

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
options.train.numEmbeddings = [20 100 200 500 1000]; % Note: 0 means "take all embeddings"
options.train.embeddingLoss = 'cosine';
options.train.embeddingLambda = [1]; % Usually, the derivatives are 1/100 of the relu derivatives right before the respective layer
options.train.embeddingMargin = [1];
% options.train.embeddingLoss = 'ACD';
% options.train.embeddingLambda = [100]; % Usually, the derivatives are 1/100 of the relu derivatives right before the respective layer
% options.train.embeddingMargin = [0.5];
options.train.samplingStrategy = {'fixed', 'fixed', 'distAware'};
options.train.samplingPartitions = {[0.5 0.5], [0.8 0.2], ''};
options.train.lu_ratio = 1; % Ratio of unlabeled to labeled data, e.g. 2 means twice the unlabeled data
options.train.fineTuneEpoch = '35';
options.train.fineTuneBaseline = [options.train.expDir_prefix '/MSBaseline/MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6/'];
options.train.fineTuneIMDBl = [IMDBPATH '/' 'imdb_msseg_domainA_c128x128_r-0_is-1_s-1_n-0_small.mat'];
options.train.fineTuneIMDBu = [IMDBPATH '/' 'imdb_msseg_domainB_c128x128_r-0_is-1_s-1_n-0_small.mat'];
options.train.repetitions = 5;
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

% Load the IMDBs for labeled and unlabeled data
imdbl = IMDB.load(options.train.fineTuneIMDBl);
imdbu = IMDB.load(options.train.fineTuneIMDBu);

% Construct the actual IMDB used for the experiment
imdb = imdbl;
imdb.images.data_unlabeled = imdbu.images.data;
imdb.images.labels_unlabeled = imdbu.images.labels;
imdb.images.lu_ratio = options.train.lu_ratio;

tic;

% Train the nets
Evals = {};
for k=1:options.train.repetitions
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
		Eval.params = struct;
		Eval.params.k = k;
		Eval.params.batchsize = options.train.batchsize;
		Eval.params.lambda = options.train.embeddingLambda(l);
		Eval.params.margin = options.train.embeddingMargin(m);
		Eval.params.samplingStrategy = vec2str(options.train.samplingPartitions{s})
		Eval.params.numEmbeddings = options.train.numEmbeddings(n);
    Evals{end+1} = Eval;
    
    end
    end
end
end
end

toc;

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
writetable(cell2table(ExcelCell), [options.train.expDir_prefix '/Report_' options.train.embeddingLoss  datestr(datetime('now'), 'HH-MM-SS-FFF') '.csv']);

% Export all average F-Scores on domain B to one Excel file
% This is a bit hardcoded, I know. But it is the fastest way to do it.
ExcelCell = {'#RUN',	'20', '100', '200', '500', '1000'};
ExcelCell{2,1} = '50-50';
ExcelCell{2 + (length(options.train.numEmbeddings) + 1),1} = '80-20';
ExcelCell{2 + 2*(length(options.train.numEmbeddings) + 1),1} = 'distAware';
for i=1:length(Evals)
	% First row is the title row. Then, the second row describes the embedding strategy. Then we have length(options.train.numEmbeddings) rows. Then the whole thing repeats.
		Eval = Evals{i};
		row_start = 2;
		if strcmp(Eval.params.samplingStrategy, '0.50.5')
			row_start = 2;
		elseif strcmp(Eval.params.samplingStrategy, '0.80.2')
			row_start = 2 + (length(options.train.numEmbeddings) + 1);
		elseif strcmp(Eval.params.samplingStrategy, '')
			row_start = 2 + 2*(length(options.train.numEmbeddings) + 1);
		end
		row = row_start + Eval.params.k;
		col = 1 + find(options.train.numEmbeddings == Eval.params.numEmbeddings);
		ExcelCell(row, 1) = {Eval.params.k};
		ExcelCell(row, col) = {(Eval.table(3).Fscore + Eval.table(4).Fscore)/2};
end
writetable(cell2table(ExcelCell), [options.train.expDir_prefix '/Report_' options.train.embeddingLoss  datestr(datetime('now'), 'HH-MM-SS-FFF') '_avgFscores.csv']);
%xlswrite([options.train.expDir_prefix '/Report_' options.train.embeddingLoss '.xlsx'], ExcelCell);

clear;