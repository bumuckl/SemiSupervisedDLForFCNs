% Creates a network architecture
%
% Author: Christoph Baur

%clear;
close all;
run ../../../Setup.m

% Options
CNN_opts;

% Override option
options.debug = true;
options.test.epoch = 'last'; % or 'last'
options.train.gpus = [];
%options.train.expDir = '../../../models/MSSEG_UNET_CWSMCELoss_DomainA_c128x128_l0.01_b6/';

% Create figure
h1 = figure;

% Load the IMDB
if ~exist('imdb', 'var')
    imdb = IMDB.load(options.data.imdbDir);
end

% Load trained model
if strcmp('last', options.test.epoch)
    options.test.epoch = num2str(CNN.findLastCheckpoint(options.train.expDir));
    epochFile = [options.train.expDir 'net-epoch-' options.test.epoch '.mat'];
else
    epochFile = [options.train.expDir 'net-epoch-' options.test.epoch '.mat'];
end
load(epochFile, 'net', 'stats') ;
net = dagnn.DagNN.loadobj(net) ;
net.mode = 'test' ;
net.removeLayer('loss');
if isnan(net.getVarIndex('prediction'))
    net.addLayer('prob', dagnn.SoftMax(), {options.net.lastLayerBeforeLoss}, {'prediction'}, {});
end
net.rebuild();
if length(options.train.gpus) > 0
    net.move('gpu');
end

% Do the test
net.mode = 'test';
for l=1:length(options.test.embeddingLayers)
    net.vars(net.getVarIndex(options.test.embeddingLayers{l})).precious = true;
end
idx_validation = find(imdb.images.set == 1);
sn = numel(idx_validation);
responses = [];
for b=1:options.test.batchsize:sn
    batch_idx = b:min(sn,b+options.test.batchsize-1);
    inputs = {'input', single(imdb.images.data(:,:,:,idx_validation(batch_idx)))} ;
    labels = imdb.images.labels(:,:,:,idx_validation(batch_idx));
    if length(options.train.gpus) > 0
        inputs{2} = gpuArray(inputs{2});
    end
    net.eval(inputs) ;
    
    % Gather classifier output
    responses = gather(net.vars(net.getVarIndex('prediction')).value);
    
    for i=1:size(responses,4)
        figure(h1), cla;
        img = squeeze(responses(:,:,2,i));
        subplot(1,2,1), imagesc(img);
        subplot(1,2,2), imagesc(squeeze(labels(:,:,:,i)));
        pause;
    end
end
