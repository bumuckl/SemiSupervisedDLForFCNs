function out = kldiv( q, p, nbins )
%KLDIV Compute the KL divergence of distribution q(x) from p(x)
%   q and p are turned into vectors
%
% Theory: The KL-Divergence is a measure of the information lost when q(x)
% is used to approximate p(x)
%
% Author: Christoph Baur <c.baur@tum.de>

% Algorithm: for q and p, determine the histograms, then compute the KL
% Divergence between the histograms

if nargin < 3
    nbins = 256;
end
epsilon = 1e-9;

[Nq,~] = histcounts(q(:),nbins);
[Np,~] = histcounts(p(:),nbins);
Nq(Nq == 0) = epsilon;
Np(Np == 0) = epsilon;

Nq = Nq ./ (sum(Nq));
Np = Np ./ (sum(Np));

out = sum( Np .* log(Np ./ Nq) );

end

