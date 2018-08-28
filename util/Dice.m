function score = Dice( A, B )
%DICE Compute dice coefficient of sets A and B
% 
% Author: Christoph Baur

    psum = sum(A(:));
    gsum = sum(B(:));
    pgsum = sum(A(:) .* B(:));
	score = (2 * pgsum) / (psum + gsum);
end

