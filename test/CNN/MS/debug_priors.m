% Compare prior and ground truth
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
tic;

% Options
CNN_opts;

% Special options
options.data.domains = {};
options.data.patients = {};
options.prior.templateSize = [5 5 5];
options.prior.numTemplatesPerDomain = 30;
options.prior.templatePatients = {'1', '2', '3'};

% Init handles
if options.debug || options.verbose
    h1 = figure;
    h2 = figure;
    h3 = figure;
end

% Load Training images & labels
images = getAllFiles(options.data.dir, '*FLAIR_preprocessed.nii.gz', true);
labelNIIs = getAllFiles(options.data.dir, '*Consensus.nii.gz', true);
priorNIIs = getAllFiles(options.data.dir, '*prior-intra.nii.gz', true);
images = [images; getAllFiles(options.data.dirMSKRI, '*rf2.nii.gz', true)];
labelNIIs = [labelNIIs; getAllFiles(options.data.dirMSKRI, '*s.nii.gz', true)];
priorNIIs = [priorNIIs; getAllFiles(options.data.dirMSKRI, '*prior-intra.nii.gz', true)];

% Status vars
numImages = length(images);
processed = 0;

% Next step: gather all testing volumes, determine their domain and compute
% a prior volume. Save the prior as a binary NII as well

for i=1:numImages
    nii_image = NII(images{i});
    nii_labels = NII(labelNIIs{i});
    nii_prior = NII(priorNIIs{i});
    
    % Normalize 
    nii_image.normalize('single');
    
    % Some hardcoded magic to distinguish different domains and patients
    patientName = hlp_getPatientName(images{i});
    patientIdx = hlp_getPatientIdx(images, patientName);
    [domainIdx,domain] = hlp_getDomain(patientName);
    
    % Visualize prior and labelmap
    labels = nii_labels.getData();
    prior = nii_prior.getData();
    [Xl, Yl, Zl] = ind2sub(size(labels), find(labels > 0));
    [Xp, Yp, Zp] = ind2sub(size(prior), find(prior > 0));
    figure(2), cla;
    %subplot(1,2,1), scatter3(Xl, Yl, Zl);
    %subplot(1,2,2), scatter3(Xp, Yp, Zp);
    [Xtp, Ytp, Ztp] = ind2sub(size(labels), find(labels .* prior));
    [Xfp, Yfp, Zfp] = ind2sub(size(labels), find((1-labels) .* prior));
    [Xfn, Yfn, Zfn] = ind2sub(size(labels), find(labels .* (1-prior)));
    subplot(1,2,1);
    scatter3(Xtp, Ytp, Ztp, 10, 'g'), hold on;
    scatter3(Xfp, Yfp, Zfp, 10, 'b'), hold on;
    scatter3(Xfn, Yfn, Zfn, 2, 'r'), hold off;
    subplot(1,2,2), scatter3(Xl, Yl, Zl);
    drawnow;
    
    % Compute Dice
    dice = Dice(double(prior), double(labels));
    disp(['Dice for patient ' num2str(patientIdx) ': ' num2str(dice)]);
end