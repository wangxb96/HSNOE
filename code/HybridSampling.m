function data = HybridSampling(data)
%%  Undersampling using NCL 
    [size1 size2]=size(data);
    Group = data(:,size2);
    adrN = find(Group==0);
    adrP = find(Group==1);
    SampleN = data(adrN,:); % Negative Training Data
    SampleP = data(adrP,:); % Positive Training Data
    size1N=size(SampleN,1);
    size1P=size(SampleP,1);
    IR = floor(size1N / size1P);
    DeleteAddress = NCL(data);  % training set
    Undersampled = data;
    if DeleteAddress ~= 0
        Undersampled(DeleteAddress,:) = [];
    end
    data = Undersampled; % after undersampling   
    [size1, size2] = size(data);
    T = sum(data(:,size2)); 
    Synthesis  = SMOTE(data,T,100 * IR,5); % SMOTE oversampling
    data = [data; Synthesis];    
    DeleteAddress = NCL(data);  % training set
    Undersampled = data;
    if DeleteAddress ~= 0
        Undersampled(DeleteAddress,:) = [];
    end
    data = Undersampled; % after undersampling  
end
