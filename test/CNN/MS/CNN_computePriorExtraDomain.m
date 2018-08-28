% Compute a prior for the testing images based on the training data and
% their labels
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

% Algorithm:
% 1. Go through all image volumes and based on the FLAIR volumes, extract
% 3D templates
% -> doing so, apply these templates to the volumes of the other domains
% and compute a prior

clear;
close all;
run ../../../Setup.m
tic;

% Options
CNN_opts;

% Special options
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
images = [images; getAllFiles(options.data.dirMSKRI, '*rf2.nii.gz', true)];
labelNIIs = [labelNIIs; getAllFiles(options.data.dirMSKRI, '*s.nii.gz', true)];

% Status vars
numImages = length(images);
processed = 0;

% Loop over all images
templates = [];
for i=1:numImages
    nii_image = NII(images{i});
    nii_labels = NII(labelNIIs{i});
    sz = nii_image.size();
    bitDepth = nii_image.nii.hdr.dime.bitpix;
    
    % Some hardcoded magic to distinguish different domains and patients
    patientName = hlp_getPatientName(images{i});
    patientIdx = hlp_getPatientIdx(images, patientName);
    [domainIdx,domain] = hlp_getDomain(patientName);
    
    if numel(options.prior.templatePatients) > 0 && sum(strcmp(options.prior.templatePatients, num2str(patientIdx))) == 0
        continue;
    end
    
    % Normalize 
    nii_image.normalize('single');
    data = nii_image.getData();
    labels = nii_labels.getData();
    
    % Determine Connected Components in current training label volume
    cc = bwconncomp(nii_labels.getData());
    S = regionprops(cc, 'Centroid', 'FilledArea');
    % Eccentricity = 0 means perfect circle!
	
    % Iterate over S and crop 3D templates if they match the desired
    % properties
    if isempty(templates)
        templates = zeros(options.prior.templateSize(1), options.prior.templateSize(2), options.prior.templateSize(3), 0);
    end
    for s=1:numel(S)
        component = S(s);
        if component.Centroid(1) < floor(options.prior.templateSize(2)/2) || ...
           component.Centroid(2) < floor(options.prior.templateSize(1)/2) || ...
           component.Centroid(3) < floor(options.prior.templateSize(3)/2) || ...
           component.Centroid(1) > size(data,2) - round(options.prior.templateSize(2)/2) || ...
           component.Centroid(2) > size(data,1) - round(options.prior.templateSize(1)/2) || ...
           component.Centroid(3) > size(data,3) - round(options.prior.templateSize(3)/2)
            continue;
        end
        %if component.FilledArea < round(0.3 * options.prior.templateSize(1) * options.prior.templateSize(2) * options.prior.templateSize(3)) && ...
        %   component.FilledArea > round(0.01 * options.prior.templateSize(1) * options.prior.templateSize(2) * options.prior.templateSize(3)) 
            ctrd = round([component.Centroid(2) component.Centroid(1) component.Centroid(3)]);
            template = data(ctrd(1)-floor(options.prior.templateSize(1)/2):ctrd(1)+floor(options.prior.templateSize(1)/2), ...
                            ctrd(2)-floor(options.prior.templateSize(2)/2):ctrd(2)+floor(options.prior.templateSize(2)/2), ...
                            ctrd(3)-floor(options.prior.templateSize(3)/2):ctrd(3)+floor(options.prior.templateSize(3)/2) );
            template_labels = labels(ctrd(1)-floor(options.prior.templateSize(1)/2):ctrd(1)+floor(options.prior.templateSize(1)/2), ...
                            ctrd(2)-floor(options.prior.templateSize(2)/2):ctrd(2)+floor(options.prior.templateSize(2)/2), ...
                            ctrd(3)-floor(options.prior.templateSize(3)/2):ctrd(3)+floor(options.prior.templateSize(3)/2) ); % TODO: crop the 3D template
            templates(:,:,:,end+1) = template;
            
            % Visualize the template
            figure(1), cla, colormap(parula(5));
            [Xl, Yl, Zl] = ind2sub(size(template_labels), find(template_labels > 0));
            scatter3(Xl, Yl, Zl);
            drawnow;
            
        %end
    end
    
    % Display process
    processed = processed + 1;
    if mod(processed, floor((0.01*numImages))) == 0
        disp(['Processed: ' num2str(processed/numImages) '%']);
    end
    
    % Clearvars
    clearvars Image;
end

% Filter templates
ridx = randperm(size(templates,4), min(size(templates,4), options.prior.numTemplatesPerDomain));
templates = templates(:,:,:,ridx);

% Save Templates
save([options.data.dir '/templatesExtraDomain.mat'], 'templates', '-v7.3');

% Next step: gather all testing volumes, determine their domain and compute
% a prior volume. Save the prior as a binary NII as well

for i=1:numImages
    nii_image = NII(images{i});
    nii_labels = NII(labelNIIs{i});
    nii_prior = NII(labelNIIs{i}); % A duplicate, but we're gonna modify it!
    
    % Normalize 
    nii_image.normalize('single');
    
    % Some hardcoded magic to distinguish different domains and patients
    patientName = hlp_getPatientName(images{i});
    patientIdx = hlp_getPatientIdx(images, patientName);
    [domainIdx,domain] = hlp_getDomain(patientName);
    
    current_templates = templates;
    tmp_priors = zeros([size(nii_image.getData()), 0]);
    for t=1:size(current_templates,4)
        template = squeeze(current_templates(:,:,:,t));
        %tmp_priors(:,:,:,end+1) = normxcorr3(template, nii_image.getData(), 'same');
        [~,tmp_priors(:,:,:,end+1)] = template_matching(template, nii_image.getData());
    end
    tmp_priors(isnan(tmp_priors(:))) = 1;
    tmp_priors = abs(tmp_priors);
    prior = geomean(tmp_priors, 4);
    
    % Visualize prior and labelmap
    options.prior.threshold = prctile(prior(:), 99.995); % Compute threshold based on percentile!
    labels = nii_labels.getData();
    [Xl, Yl, Zl] = ind2sub(size(labels), find(labels > 0));
    [Xp, Yp, Zp] = ind2sub(size(prior), find(prior > options.prior.threshold));
    figure(2), cla;
    subplot(1,2,1), scatter3(Xl, Yl, Zl);
    subplot(1,2,2), scatter3(Xp, Yp, Zp);
    drawnow;
    
    % Set prior NII and save it
    [pathstr,~,~] = fileparts(images{i});
    filename_prior = [pathstr '/prior-extra.nii'];
    nii_prior.setData(prior > options.prior.threshold);
%     nii_prior.save(filename_prior);
    gzip(filename_prior);
    delete(filename_prior);
    
    % Compute Dice
    if numel(options.prior.templatePatients) > 0 && sum(strcmp(options.prior.templatePatients, num2str(patientIdx))) == 0
        dice = Dice(double(nii_prior.getData()), double(labels));
        disp(['Dice for patient ' num2str(patientIdx) ': ' num2str(dice)]);
    end
end