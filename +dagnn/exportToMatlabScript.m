function exportToMatlabScript( net, target )
%EXPORTTOMATLABSCRIPT Given a net matlab object of type DAGNN, create a .m
%file which creates such a network from scratch
%
% Author: Christoph Baur

    % Helper vars
    endline = '\n';
    
    % Output string variable
    str = ['net = dagnn.DagNN();' endline];
    
    % Go Go Go
    layers = net.layers;
    for l=1:length(layers)
       block = ['block = ' layers(l).type '(' hlp_structToKeyValuePairs(layers(l).block) ');' endline];
       
       addblock = ['net.addLayer(' '''' layers(l).name ''', block, '];
       
       addblock = [addblock '{'];
       for i=1:length(layers(l).inputs)
           if i > 1
               addblock = [addblock ', '];
           end
           addblock = [addblock '''' layers(l).inputs{i} ''''];
       end
       addblock = [addblock '}, '];
       
       addblock = [addblock '{'];
       for i=1:length(layers(l).outputs)
           if i > 1
               addblock = [addblock ', '];
           end
           addblock = [addblock '''' layers(l).outputs{i} ''''];
       end
       addblock = [addblock '}, '];
       
       addblock = [addblock '{'];
       for i=1:length(layers(l).params)
           if i > 1
               addblock = [addblock ', '];
           end
           addblock = [addblock '''' layers(l).params{i} ''''];
       end
       addblock = [addblock '}'];
       
       addblock = [addblock ');' endline];
       
       str = [str block addblock];
    end
    
    str = [str 'net.initParams();'];
    
    % Write Out
    fid = fopen(target, 'wt');
    fprintf(fid, str);
    fclose(fid);
end

function str = hlp_structToKeyValuePairs(s)
    names = fieldnames(s);
    str = '';
    for i=1:length(names)
        if strcmp(names{i}, 'opts')
            continue;
        end
        if i > 1
            str = [str ', '];
        end
        str = [str '''' names{i} ''', [' sprintf(' %d',s.(names{i})) ']'];
    end
end

