classdef HTMLReport < handle
    %HTMLREPORT Creates a html document with tables and stuff for comparing
    %different models
    %
    % Author: Christoph Baur
    
    properties
        folder = '';
        blocks = {};
        rows = {};
        title = '';
        description = '';
        targetDir = '';
    end
    
    methods
        function obj = HTMLReport(title, description)
            obj.title = title;
            obj.description = description;
            [obj.folder, ~, ~] = fileparts(which('HTMLReport'));
        end
        
        function rowNumber = addRow(self, numCols)
            row = struct;
            row.cols = {};
            
            self.rows{end+1} = row;
            rowNumber = length(self.rows); 
        end
        
        function addCol(self, title, elements, rowNumber)
            col = struct;
            col.title = title;
            col.elements = elements;
            
            self.rows{rowNumber}.cols{end+1} = col;
        end
        
        function save(self, dirname)
            self.targetDir = dirname;
            copyfile([self.folder '/resources'], dirname)
            
            fin = fopen([self.folder '/resources/index.html'], 'r');
            fout = fopen([dirname '/index.html'], 'w');
            
            while ~feof(fin)
                s = fgetl(fin);
                s = strrep(s, '{yield}', self.getHTML());
                s = strrep(s, '{title}', self.title);
                s = strrep(s, '{description}', self.description);
                fprintf(fout,'%s',s);
                disp(s);
            end
            
            fclose(fin);
            fclose(fout);
            self.targetDir = '';
        end
        
        % Internal
        
        function html = getHTML(self)
            html = '';
            
            for r=1:length(self.rows)
               row = self.rows{r};
               
               html = [html '<div class="row">'];
               
               colWidth = 12 / length(row.cols);
               for c=1:length(row.cols)
                   col = row.cols{c};
                   
                   html = [html '<div class="col-md-' num2str(colWidth) '">'];
                   html = [html '<h2>' col.title '</h2>'];
                   
                   for e=1:length(col.elements)
                       element = col.elements{e};
                       if strcmp(element.type, 'image')
                           % Copy image to the report folder
                           try
                               [pathstr, name, ext] = fileparts(element.path);
                               fileinfo = dir(element.path);
                               uniqueFilename = [name '-' num2str(fileinfo.datenum) ext];
                               absImagePath = [self.targetDir '/img/' uniqueFilename];
                               relativeImagePath = ['./img/' uniqueFilename];
                               copyfile(element.path, [self.targetDir '/img/' uniqueFilename]);
                           catch exc
                               relativeImagePath = '';
                           end
                           html = [html '<img src="' relativeImagePath '" class="img-responsive" alt="" />'];
                           html = [html '<legend>' element.caption '</legend>'];
                       elseif strcmp(element.type, 'pdf')
                           % Copy PDF to the report folder
                           try
                               [pathstr, name, ext] = fileparts(element.path);
                               fileinfo = dir(element.path);
                               uniqueFilename = [name '-' num2str(fileinfo.datenum) ext];
                               absImagePath = [self.targetDir '/img/' uniqueFilename];
                               relativeImagePath = ['./img/' uniqueFilename];
                               copyfile(element.path, [self.targetDir '/img/' uniqueFilename]);
                           catch exc
                               relativeImagePath = '';
                           end
                           html = [html '<object data=' relativeImagePath '"></object>'];
                           html = [html '<legend>' element.caption '</legend>'];
                       elseif strcmp(element.type, 'text')
                           html = [html '<p>' element.text '</p>'];
                       elseif strcmp(element.type, 'heading')
                           html = [html '<h3>' element.text '</h3>'];
                       elseif strcmp(element.type, 'keyValue')
                           html = [html '<p>'];
                           for k=1:length(element.keys)
                           html = [html element.keys{k} ': ' element.values{k} '<br>'];
                           end
                           html = [html '</p>'];
                       elseif strcmp(element.type, 'table')
                           html = [html '<table class="table table-striped">'];
                           headings = fieldnames(element.table);
                           if numel(headings) > 0
                               html = [html '<thead><tr>'];
                                    for h=1:numel(headings)
                                        html = [html '<th>' headings{h} '</th>'];
                                    end
                               html = [html '</tr></thead>'];
                           end
                           % Iterate over every row
                           html = [html '<tbody>'];
                           for tr=1:size(element.table,2)
                               html = [html '<tr>'];
                               for td=1:numel(headings)
                                  html = [html '<td>' asString(element.table(tr).(headings{td})) '</td>'];
                               end
                               html = [html '</tr>'];
                           end
                           html = [html '</tbody></table>'];
                       else
                           error('HTMLReport: Invalid element type');
                       end
                   end
                   
                   html = [html '</div>'];
               end
               
               html = [html '</div>'];
            end

        end
        
        % HELPERS
        
    end
    
    methods(Static)
        function element = pdfElement(path, caption)
            element = struct;
            element.type = 'pdf';
            element.path = path;
            element.caption = caption;
        end
        
        function element = imageElement(path, caption)
            element = struct;
            element.type = 'image';
            element.path = path;
            element.caption = caption;
        end
        
        function element = textElement(text)
            element = struct;
            element.type = 'text';
            element.text = text;
        end
        
        function element = headingElement(text)
            element = struct;
            element.type = 'heading';
            element.text = text;
        end
        
        function element = keyValueElement(keys, values)
            element = struct;
            element.type = 'keyValue';
            element.keys = keys;
            element.values = values;
        end
        
        function element = tableElement(matlabStruct) % Expects a one-or multirow struct
            element = struct;
            element.type = 'table';
            element.table = matlabStruct;
        end
        
    end
    
end

function str = asString(input)
    if isnumeric(input)
        str = num2str(input);
    else
        str = string(input);
    end
end
