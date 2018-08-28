% Given the path to the models, create a HTML Report of all models
%
% Author: Christoph Baur

clear;
close all;
run ../../../Setup.m

% Options
CNN_opts;

% Override options
options.report.baselineModel = '../MNIST3_p28x28_l0.0001_b100/';
options.report.modelPath = [options.train.expDir_prefix '/semisupervisedEmbeddingLossSimilarity'];
options.report.match = '';
options.report.strip = 'MNIST3_';
options.report.epoch = '10';
options.report.title = '10-class SemiSupervisedEmbeddingLoss';
options.report.description = 'Additionally to a 10-class softmax classifier, add a semi-supervised embedding loss to embed similar patches (based on a perfect prior).';
options.report.rowKey = 'data_fraction';
options.report.expDir = [options.train.expDir_prefix '/reports/semisupervisedEmbeddingLossSimilarity'];
options.test.embeddingType = '10class';
options.test.epoch = 'last'; % or 'last'
options.test.tsne = false;
options.test.pca = true;
options.test.savefig = true;

% Init vars
params = struct('name',{},'type',{},'data_fraction',{},'patchsize',{},'batchsize',{},'learning_rate',{},'lambda',{},'margin',{},'lu',{},'embeddingLayers',{})
evals = {};

% Algorithm:
% In the modelPath, list all folders
% Filter the folders for the ones that contain the substring defined in
% options.report.match
% Then, for every folder, extract the parameters from the folder name and
% load the Eval.mat
% Now we can build the actual HTML report

% List all models in modelPath
d = dir(options.report.modelPath);
isub = [d(:).isdir];
models = {d(isub).name}';
models(ismember(models,{'.','..'})) = [];

% Filter the models for the ones that match options.report.match
if length(options.report.match) > 0
    models(cellfun('isempty', strfind(models, options.report.match))) = [];
end

% Init report
report = HTMLReport(options.report.title, options.report.description);

% Iterate over all models and fill modelTable
for m=0:length(models)
    idx = length(params)+1;
    if m == 0
        model = options.report.baselineModel;
        params(idx).data_fraction = '1';
    else
        model = models{m};
    end
    if ~exist([options.report.modelPath '/' model '/net-epoch-' options.report.epoch '.mat'], 'file')
        continue;
    end
    if ~exist([options.report.modelPath '/' model '/eval-epoch-' options.report.epoch], 'dir')
        options.train.expDir = [options.report.modelPath '/' model '/'];
        fn_test(options);
    end
    
    % Load Eval.mat
    load([options.report.modelPath '/' model '/eval-epoch-' options.report.epoch '/Eval.mat']);
    evals{idx} = Eval;
    
    % Extract model name
    model_stripped = strrep(model, options.report.strip, '');
    
    % Extract parameters from model name
    parts = strsplit(model_stripped, '_');
    params(idx).name = model;
    params(idx).embeddingLayers = {};
    for p=1:length(parts)
        if length(strfind(parts{p}, 'semisupervised'))
            params(idx).type = 'semisupervised';
        elseif length(strfind(parts{p}, 'supervised'))
            params(idx).type = 'supervised';
        elseif length(strfind(parts{p}, 'embeddingLoss'))
            params(idx).type = 'embeddingLoss';
        elseif length(strfind(parts{p}, 'lambda'))
            params(idx).lambda = strrep(parts{p}, 'lambda', '');
        elseif length(strfind(parts{p}, 'lu'))
            params(idx).lu = strrep(parts{p}, 'lu', '');
        elseif length(strfind(parts{p}, 'l'))
            params(idx).learning_rate = strrep(parts{p}, 'l', '');
        elseif length(strfind(parts{p}, 'p'))
            params(idx).patchsize = strrep(parts{p}, 'p', '');
        elseif length(strfind(parts{p}, 'b'))
            params(idx).batchsize = strrep(parts{p}, 'b', '');
        elseif length(strfind(parts{p}, 'f'))
            params(idx).data_fraction = strrep(parts{p}, 'f', '');
        elseif length(strfind(parts{p}, 'm'))
            params(idx).margin = strrep(parts{p}, 'm', '');
        else
            params(idx).embeddingLayers{end+1} = parts{p};
        end
    end

end

% Now go through the modelTable by the rowKey and create the report
rowKeyValues = extractfield(params, options.report.rowKey);
rows = unique(rowKeyValues);
for r=1:length(rows)
   subIdx = find(not(cellfun('isempty', strfind(rowKeyValues, rows{r}))));
   submodels = params(1, subIdx);
   subevals = evals(subIdx);
   
   rowId = report.addRow(length(submodels));
   
   for c=1:length(subevals)
       % Build elements of this block
       elements = {};
       elements{end+1} = HTMLReport.imageElement([options.report.modelPath '/' submodels(c).name '/net-train.pdf'], 'Learning Curves');
       elements{end+1} = HTMLReport.imageElement([options.report.modelPath '/' submodels(c).name '/eval-epoch-' options.report.epoch '/embedding_pca_FC1.png'], 'FC1 Embeddings via PCA');
       elements{end+1} = HTMLReport.imageElement([options.report.modelPath '/' submodels(c).name '/eval-epoch-' options.report.epoch '/embedding_pca_FC2.png'], 'FC2 Embeddings via PCA');
       
       elements{end+1} = HTMLReport.headingElement('Params');
       keys = {'Type', 'Data fraction', 'Patchsize', 'Batchsize', 'Learning rate', 'lambda', 'margin', 'lu_ratio'};
       values = {submodels(c).type, submodels(c).data_fraction, submodels(c).patchsize, submodels(c).batchsize, submodels(c).learning_rate, submodels(c).lambda, submodels(c).margin, submodels(c).lu};
       elements{end+1} = HTMLReport.keyValueElement(keys, values);
       
       elements{end+1} = HTMLReport.headingElement('Evaluation');
       keys = {'Classification Error'};
       values = {num2str(subevals{c}.classification_error)};
       elements{end+1} = HTMLReport.keyValueElement(keys, values);
       try
       keys = {'Clustering (Davies-Bouldin) FC1', 'Clustering (Davies-Bouldin) FC2'};
       values = {num2str(subevals{c}.clusterQualityDaviesBouldin{1}.CriterionValues), num2str(subevals{c}.clusterQualityDaviesBouldin{2}.CriterionValues)};
       elements{end+1} = HTMLReport.keyValueElement(keys, values);
       end
       try
       keys = {'Clustering (Silhouette) FC1', 'Clustering (Silhouette) FC2'};
       values = {num2str(subevals{c}.clusterQualitySilhouette{1}.CriterionValues), num2str(subevals{c}.clusterQualitySilhouette{2}.CriterionValues)};
       elements{end+1} = HTMLReport.keyValueElement(keys, values);
       end
       elements{end+1} = HTMLReport.textElement(submodels(c).name);
       
       report.addCol(num2str(c), elements, rowId);
   end
end

if ~exist(options.report.expDir)
    mkdir(options.report.expDir);
end

report.save(options.report.expDir);