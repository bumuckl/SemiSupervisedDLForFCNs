function [ histo ] = histogram2( signal )

    bins = unique( signal );
    histo = zeros([1 numel(bins)]);
    for i = 1:numel(bins)
        histo(i) = sum(signal == bins(i));
    end
end
