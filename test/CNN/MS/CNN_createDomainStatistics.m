% Given the path to the models, create a CSV file of average Precision,
% Recall and F-Score per Domain (and overall)
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

% Options
CNN_opts;

% Override options
options.report.modelPath = [options.train.expDir_prefix '/MSSemiSupervised/'];
options.report.epoch = 50;

% Find all models inside the modelDir
d = dir(options.report.modelPath);
isub = [d(:).isdir];
models = {d(isub).name}';

% Iterate over all models, compute the stats per Domain and output the csv
% file
for m=3:length(models)
    model = [options.report.modelPath '/' models{m}];
    evalFolder = [model '/eval-epoch-' num2str(options.report.epoch)];
    if ~exist(evalFolder, 'dir')
        continue;
    end
    if ~exist([evalFolder '/Eval.mat'], 'file')
        continue;
    end
    
    % Load Eval.mat, then we have a variable called "Eval"
    load([model '/eval-epoch-' num2str(options.report.epoch) '/Eval.mat']);
    
    % Per Domain & Overall
    alldomains = [];
    for r=1:size(Eval.table,2)
        alldomains = [alldomains Eval.table(r).domain];
    end
    domains = unique(alldomains);
    Eval.all.avgfscore = 0;
    Eval.all.avgprecision = 0;
    Eval.all.avgrecall = 0;
    Eval.all.avgerror = 0;
    Eval.all.numNan = 0;
    for d=1:numel(domains)
        domain_idx = find(alldomains == domains(d));
        Eval.domains(d).avgfscore = 0;
        Eval.domains(d).avgprecision = 0;
        Eval.domains(d).avgrecall = 0;
        Eval.domains(d).avgerror = 0;
        Eval.domains(d).numNan = 0;
        for i=1:numel(domain_idx)
            Eval.domains(d).avgprecision = Eval.domains(d).avgprecision + Eval.table(domain_idx(i)).Precision;
            Eval.domains(d).avgrecall = Eval.domains(d).avgrecall + Eval.table(domain_idx(i)).Recall;
            Eval.domains(d).avgerror = Eval.domains(d).avgerror + Eval.table(domain_idx(i)).error;
            if ~isnan(Eval.table(domain_idx(i)).fscore)
                Eval.domains(d).avgfscore = Eval.domains(d).avgfscore + Eval.table(domain_idx(i)).fscore;
            else
                Eval.domains(d).numNan = Eval.domains(d).numNan + 1;
            end
        end
        Eval.all.avgfscore = Eval.all.avgfscore + Eval.domains(d).avgfscore;
        Eval.all.avgprecision = Eval.all.avgprecision + Eval.domains(d).avgprecision;
        Eval.all.avgrecall = Eval.all.avgrecall + Eval.domains(d).avgrecall;
        Eval.all.avgerror = Eval.all.avgerror + Eval.domains(d).avgerror;
        Eval.all.numNan = Eval.all.numNan + Eval.domains(d).numNan;
        Eval.domains(d).domain = domains(d);
        Eval.domains(d).avgfscore = Eval.domains(d).avgfscore / (numel(domain_idx) - Eval.domains(d).numNan);
        Eval.domains(d).avgprecision = Eval.domains(d).avgprecision / numel(domain_idx);
        Eval.domains(d).avgrecall = Eval.domains(d).avgrecall / numel(domain_idx);
        Eval.domains(d).avgerror = Eval.domains(d).avgerror / numel(domain_idx); 
    end
    Eval.all.avgfscore = Eval.all.avgfscore / (size(Eval.table,2) - Eval.all.numNan);
    Eval.all.avgprecision = Eval.all.avgprecision / size(Eval.table,2);
    Eval.all.avgrecall = Eval.all.avgrecall / size(Eval.table,2);
    Eval.all.avgerror = Eval.all.avgerror / size(Eval.table,2);
    
    % Export CSV
    % Fscore
    T = table(  Eval.domains(1).avgfscore, Eval.domains(2).avgfscore, ...
                Eval.domains(3).avgfscore, Eval.domains(4).avgfscore, ...
                Eval.all.avgfscore, ...
                'VariableNames', {'FscoreA', 'FscoreB', 'FscoreC', 'FscoreD', 'FscoreAll'} ...
            );
    writetable(T,[evalFolder '/avgfscores.csv']);
    % Precision
    T = table(  Eval.domains(1).avgprecision, Eval.domains(2).avgprecision, ...
                Eval.domains(3).avgprecision, Eval.domains(4).avgprecision, ...
                Eval.all.avgprecision, ...
                'VariableNames', {'PrecisionA', 'PrecisionB', 'PrecisionC', 'PrecisionD', 'PrecisionAll'} ...
            );
    writetable(T,[evalFolder '/avgprecisions.csv']);
    % Recall
    T = table(  Eval.domains(1).avgrecall, Eval.domains(2).avgrecall, ...
                Eval.domains(3).avgrecall, Eval.domains(4).avgrecall, ...
                Eval.all.avgrecall, ...
                'VariableNames', {'RecallA', 'RecallB', 'RecallC', 'RecallD', 'RecallAll'} ...
            );
    writetable(T,[evalFolder '/avgrecalls.csv']);
end