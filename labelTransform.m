for i = 1 : 801 
    if isequal(label{i},'BRCA') == 1 label{i} = 1; 
    elseif isequal(label{i},'COAD') == 1 label{i} = 2; 
    elseif isequal(label{i},'KIRC') == 1 label{i} = 3;
    elseif isequal(label{i},'LUAD') == 1 label{i} = 4;
    elseif isequal(label{i},'PRAD') == 1 label{i} = 5;
    end
end