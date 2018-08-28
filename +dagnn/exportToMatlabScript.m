function exportToMatlabScript( net, target )
%EXPORTTOMATLABSCRIPT Given a net matlab object of type DAGNN, create a .m
%file which creates such a network from scratch
%
% Copyright (c) 2016-2017, Christoph Baur <c.baur@tum.de>. All rights reserved.
%
% This work is licensed under the Creative Commons Attribution-NonCommercial 
% 4.0 International License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

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

