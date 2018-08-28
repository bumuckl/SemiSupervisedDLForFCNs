function patch = patchAroundPoint( I, point, patchsize )
%PATCHAROUNDPOINT Given an image, a point and a patchsize, extract a patch
%around that point
% I: the input image
% point: a 2-element vector in the format of [y x]
% patchsize: a 2-element vector of the size of each patch in [h w]
%
% @Author: Christoph Baur

    patchsize_half = floor(patchsize ./ 2);

    if point(1) <= patchsize_half(1) || point(1) > size(I,1) - patchsize_half(1) || point(2) <= patchsize_half(2) || point(2) > size(I,2) - patchsize_half(2)
        error('patchAroundPoint: point outside valid region!');
    end

	if mod(patchsize(1),2) == 0
		%number is even
		patch = I(point(1)-patchsize_half(1):point(1)+patchsize_half(1)-1, ...
              point(2)-patchsize_half(2):point(2)+patchsize_half(2)-1, :);
	else
		%number is odd
		patch = I(point(1)-patchsize_half(1):point(1)+patchsize_half(1), ...
              point(2)-patchsize_half(2):point(2)+patchsize_half(2), :);
	end
end

