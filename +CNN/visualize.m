function handle = visualize( net, savepath )
%VISUALIZE Given a net, visualize its filters

	handle = figure;
	
	% First round: find out dimensions of subplot that is going to be built. Therefore, iterate over the complete net and count all filters
	maxFilters = 0;
	numLayers = length(net.layers);
	for i=1:numLayers
		if ~strcmp(net.layers{i}.type, 'conv')
			continue;
		end
		maxFilters = max(size(net.layers{i}.filters, 4), maxFilters);
	end
	
	% Now plot the whole thing
	figure(handle), hold on;
	for l=1:numLayers
		if ~strcmp(net.layers{l}.type, 'conv')
			continue;
		end
		for f=1:size(net.layers{l}.filters, 4)
			subplot(maxFilters, numLayers, maxFilters*(l-1) + f), imshow(squeeze(net.layers{l}.filters(:,:,:,f))), hold on;
		end
	end
	
	if nargin > 1
		savefig(handle, savepath);
	end
end

