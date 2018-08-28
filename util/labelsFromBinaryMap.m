function labels = labelsFromBinaryMap( map )
%LABELSFROMBINARYMAP Given a binary map, return the locations of white
%pixels as labels. Each row in labels corresponds to [y x]

    ind = find(map);
    [r, c] = ind2sub(size(map), ind);
    labels = [r c];
end

