function response = classifyBatch( net, data )
%CLASSIFYBATCH Given a batch of images, classify them all at once and
%return a matrix of responses

    res = vl_simplenn(net, data, [], [], 'disableDropout', true);

    response = [];
    for i=1:size(data,4)
       response(:,i) = res(end).x(1,1,:,i);
    end
end

