function out = structarrayavg( sarr, fieldindex )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    tmp = squeeze(struct2cell(sarr));
    tmp = tmp(fieldindex,:);
    tmp = cell2mat(tmp);
    tmp_sum = nansum(tmp);
    nonnanidx = find(~isnan(tmp));
    
    tmp = tmp(nonnanidx);
    
    out = struct;
    out.mean = mean(tmp);
    out.std = std(tmp);
    out.median = median(tmp);
    out.min = min(tmp);
    out.max = max(tmp);
end

