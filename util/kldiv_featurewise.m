function out = kldiv_featurewise( q, p, nbins )
%KLDIV Compute the KL divergence of distribution q(x) from p(x)
%   q and p are matrices whose columns are samples and rows are features.
%   the kl is computed for every feature
%
% Theory: The KL-Divergence is a measure of the information lost when q(x)
% is used to approximate p(x)
%
% Author: Christoph Baur <c.baur@tum.de>

% Algorithm: for each feature in q and p, determine the histograms, then compute the KL
% Divergence between the histograms

if nargin < 3
    nbins = 256;
end

numfeatures = size(q,1);
out = zeros(numfeatures,1);
for i=1:numfeatures
    out(i) = kldiv(q(i,:), p(i,:));
end

end

