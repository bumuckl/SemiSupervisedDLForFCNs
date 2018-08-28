function fig = cnn_plot_curve_at_epoch( modelPath, epoch )

    modelPath = fullfile(modelPath, ['net-epoch-' num2str(epoch) '.mat']);
    modelFigPath = fullfile(modelPath, ['net-train-' num2str(epoch) -'.pdf']) ;

    [net, stats] = loadState(modelPath(start)) ;
    
    fig = figure;
    plots = setdiff(...
      cat(2,...
      fieldnames(stats.train)', ...
      fieldnames(stats.val)'), {'num', 'time'}) ;
    for p = plots
      p = char(p) ;
      values = zeros(0, epoch) ;
      leg = {} ;
      for f = {'train', 'val'}
        f = char(f) ;
        if isfield(stats.(f), p)
          tmp = [stats.(f).(p)] ;
          values(end+1,:) = tmp(1,:)' ;
          leg{end+1} = f ;
        end
      end
      subplot(1,numel(plots),find(strcmp(p,plots))) ;
      plot(1:epoch, values','o-') ;
      xlabel('epoch') ;
      title(p) ;
      legend(leg{:}) ;
      grid on ;
    end
    drawnow ;
    print(1, modelFigPath, '-dpdf') ;

end

