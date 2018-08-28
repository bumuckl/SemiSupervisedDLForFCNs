function responseMap = responseMapFromResponses( responses, positions, imagesize )
%RESPONSEMAPFROMRESPONSES Given a vector of responses and their 2d
%prositions [y,x], as well as the desired mapsize, create the corresponding
%responseMap

    responseMap = zeros(imagesize(1), imagesize(2));
    for p=1:size(positions,1)
        responseMap(positions(p,1), positions(p,2)) = responses(p);
    end
end

