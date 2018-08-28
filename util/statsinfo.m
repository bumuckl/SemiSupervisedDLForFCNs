function statsinfo( matrix )
    disp(['Min: ' num2str(min(matrix(:))) ' Max: ' num2str(max(matrix(:))) ' Unique: ' num2str(numel(unique(matrix(:))))]);
end

