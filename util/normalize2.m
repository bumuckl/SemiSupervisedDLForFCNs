function I_norm = normalize2( I, precision, bitdepth )
%NORMALIZE Given an input image, make sure its intensity values reside
%within the range of [0;1].
%
% @Author: Christoph Baur

    if nargin < 2
        precision = 'double';
    end
    
    if strcmp(precision, 'double')
        I_norm = double(I);
    elseif strcmp(precision, 'single')
        I_norm = single(I);
    else
        I_norm = double(I);
    end
    
    I_norm = I_norm - min(min(I_norm(:)), 0);
    
    if nargin < 3
        I_norm = I_norm ./ max(I_norm(:));
    else
        I_norm = I_norm ./ ((2^bitdepth)-1);
    end
end

