function data = HybridSampling(data)
%%  Undersampling using NCL 
    DeleteAddress = NCL(data);  % training set
    Undersampled = data;
    if DeleteAddress ~= 0
        Undersampled(DeleteAddress,:) = [];
    end
    data = Undersampled; % after undersampling   
    [size1, size2] = size(data);
    T = sum(data(:,size2)); 
    Synthesis  = SMOTE(data,T,100,5); % SMOTE oversampling
    data = [data; Synthesis];    
    DeleteAddress = NCL(data);  % training set
    Undersampled = data;
    if DeleteAddress ~= 0
        Undersampled(DeleteAddress,:) = [];
    end
    data = Undersampled; % after undersampling  
    % train = data;
    % save('test100.mat','train');

    %     X = data(:,1:end-1);
    %     Y = data(:, end);
    %     cv = cvpartition(Y,'holdout',0.1);
    %     indx = cv.test;
    %     Xtrain = X(~indx,:);
    %     Ytrain = Y(~indx);
    %     Xtest = X(indx,:);
    %     Ytest = Y(indx);   
    
    %     train = [Xtrain,Ytrain]; %For Training
    %     save('train100.mat','train');
    %     test = [Xtest,Ytest]; %For Test
    %     save('test100.mat','test');
end