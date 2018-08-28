function out = jsdiv( q, p, nbins )
%JSDIV Compute the Jensen-Shannon divergence of distribution q(x) and p(x)
%   q and p are turned into vectors
%
% Author: Christoph Baur <c.baur@tum.de>

% Algorithm: for q and p, determine the histograms, then compute the JS
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

M = 0.5.*(Np + Nq);

kldivpm = sum( Np .* log(Np ./ M) );
kldivqm = sum( Nq .* log(Nq ./ M) );
out = 0.5.*kldivpm + 0.5*kldivqm;

end

