function dagnnToM( dagnn )
%DAGNNTOM Given a dagnn object, convert it to the contents of a .m file
%which would have created it.
%
% Author: Christoph Baur

    str = '';
    newline = char(10);
    for i=1:length(dagnn.layers)
        % Build the layer - Code
        str = ['block = ' dagnn.layers(i).type '('];
        if strcmp(dagnn.layers(i).type, 'dagnn.Conv')
            str = [str '''size'', ' hlp_vec2string(dagnn.layers(i).block.size) ...
                   ', ''hasBias'', ' num2str(dagnn.layers(i).block.hasBias) ...
                   ', ''pad'', ' hlp_vec2string(dagnn.layers(i).block.pad) ...
                   ', ''stride'', ' hlp_vec2string(dagnn.layers(i).block.stride) ...
                  ];
        elseif strcmp(dagnn.layers(i).type, 'dagnn.BatchNorm')
            str = [str '''numChannels'', ' hlp_vec2string(dagnn.layers(i).block.numChannels) ...
                   ', ''epsilon'', ' num2str(dagnn.layers(i).block.epsilon) ...
                  ];
        elseif strcmp(dagnn.layers(i).type, 'dagnn.Pooling')
            str = [str '''method'', ''' dagnn.layers(i).block.method ...
                   ''', ''poolSize'', ' hlp_vec2string(dagnn.layers(i).block.poolSize) ...
                   ', ''pad'', ' hlp_vec2string(dagnn.layers(i).block.pad) ...
                   ', ''stride'', ' hlp_vec2string(dagnn.layers(i).block.stride) ...
                  ];
        end
        str = [str ');' newline];
        
        % Add it - Code
        str = [str 'net.addLayer(' '''' dagnn.layers(i).name ''', block, ' ...
               hlp_cell2string(dagnn.layers(i).inputs) ', ' ...
               hlp_cell2string(dagnn.layers(i).outputs) ', ' ...
               hlp_cell2string(dagnn.layers(i).params) ...
               ');'];
        str = [str newline];
        
        disp(strjoin(str,''));
    end
end

function str = hlp_vec2string(vec)
    str = '';
    for i=1:length(vec)
        if i==1
            str = [str num2str(vec(i))];
        else
            str = [str ' ' num2str(vec(i))];
        end
    end
    
    str = ['[' str ']'];
end

function str = hlp_cell2string(cll)
    str = '';
    
    if length(cll) > 0
    for i=1:length(cll)
        if i==1
            str = [str '''' cll(i) ''''];
        else
            str = [str ', ' '''' cll(i) ''''];
        end
    end
    end
    
    str = ['{' str '}'];
end

