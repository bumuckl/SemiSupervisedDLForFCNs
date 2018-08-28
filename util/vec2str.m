function str = vec2str( vec )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    str = '';
    for i=1:numel(vec(:))
        if i == 1
            str = num2str(vec(i));
        else
            str = [str num2str(vec(i))];
        end
    end
end

