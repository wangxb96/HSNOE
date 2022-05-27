addpath(genpath('ParameterAnalysis'));
numRun = 1;
for l=1:numRun
    warning('off','all');
    %% Model SETTINGS
    params.classifiers = {'KNN'};%, 'DT', 'DISCR', 'NB', 'ANN'};
    
    %% MAIN LOOP 
    p_name = {'Data'};
    results.p_name = p_name;        

    %% Hybrid Sampling & Data Partition 
%     data = importdata('DATA.mat');
%     %%  Undersampling using NCL 
%     DeleteAddress = NCL(data);  % training set
%     Undersampled = data;
%     if DeleteAddress ~= 0
%         Undersampled(DeleteAddress,:) = [];
%     end
%     data = Undersampled; % after undersampling   
%     [size1, size2] = size(data);
%     T = sum(data(:,size2)); 
%     Synthesis  = SMOTE(data,T,100,5); % SMOTE oversampling
%     data = [data; Synthesis];    
%     DeleteAddress = NCL(data);  % training set
%     Undersampled = data;
%     if DeleteAddress ~= 0
%         Undersampled(DeleteAddress,:) = [];
%     end
%     data = Undersampled; % after undersampling  
%     
%     X = data(:,1:end-1);
%     Y = data(:, end);
%     cv = cvpartition(Y,'holdout',0.1);
%     indx = cv.test;
%     Xtrain = X(~indx,:);
%     Ytrain = Y(~indx);
%     Xtest = X(indx,:);
%     Ytest = Y(indx);   
%     
%     train = [Xtrain,Ytrain]; %For Training
%     save('train100.mat','train');
%     test = [Xtest,Ytest]; %For Test
%     save('test100.mat','test');

    %% Import Data (train & test)
    train = importdata('train100.mat');
    test = importdata('test100.mat');
    
%     data = importdata('NCLTrain.mat');
%     tdata = importdata('NCLTest.mat');      
    data = train;
    testdata = test;

    feat = data(:,1:end-1);
    label = data(:,end);
    Acc0 = jknn(data(:,1:end-1),data(:,end));
   
    
   %% basic settings of ACO algorithm
    num = 5;
    opts.N  = 100;     % number of solutions 100
    opts.T  = 50;    % maximum number of iterations
    opts.num = num; 
    
   %% Feature selection in each cluster by ACO algorithm
    fun = @jFitnessFunction; %set the objective function 
    ACO = jAntColonyOptimization(fun,data(:,1:end-1),data(:,end),opts);
    sf = ACO.sf;
    tdata = [data(:,sf), data(:,end)];        
    testdata = [testdata(:,sf),testdata(:,end)];     
    test = [test(:,sf),test(:,end)];  

%     tdata = data;
    finalClassifiers = [];
    cvs = cvpartition(tdata(:,end), 'KFold', 5);
    for f = 1 : 5
        idxs = cvs.test(f);
        testData = tdata(idxs,:);
        trainData = tdata(~idxs, :);
        
        train = trainData;
        validation = testData;

        trainX = train(:, 1:end-1);
        trainy = train(:, end);

        valX = validation(:, 1:end-1);
        valy = validation(:, end);

        trainX(isnan(trainX)) = -1;
        valX(isnan(valX)) = -1;

        allClusters = generateClustersV2([trainX, trainy], params);
        [allClusters, centroids] = balanceClusters(allClusters, [trainX trainy]);
        for i = 1 : length(allClusters)
            balancedClusters{i} = allClusters{i};
%             balancedClusters{i} = allClusters{i}.train;
        end
        classifierIndex = 1;
        for c=balancedClusters
            X = c{1,1}(:, 1:end-1);
            y = c{1,1}(:, end);
%         for c = 1 : length(balancedClusters)
%             X = balancedClusters{c}(:,1:end-1);
%             y = balancedClusters{c}(:,end);
            all = trainClassifiers(X, y, params);
            if size(all,1) < 1
                continue
            end
            for temp = 1:length(all)
                classifiers{classifierIndex} = all{1,temp};
                classifierIndex = classifierIndex + 1;
            end 
        end
        [~,pred] = fusion(classifiers,validation);
        SC = [];
        acc = zeros(1,size(pred,2));
        for i = 1 : size(pred,2)
            acc(i) = mean(pred(:,i) == validation(:,end));
        end    
        for i = 1 : size(pred,2)
            if acc(i) >= mean(acc)
               SC = [SC, i];
            end
        end
        classifiers = classifiers(:,SC);
        ACO = classifierSelectionACO(classifiers, validation, opts);
        auc01(f) = fusion(classifiers(ACO.sc), testData);
        auc02(f) = fusion(classifiers, testData);
        disp(auc01(f)); disp(auc02(f));
        classifiers = classifiers(ACO.sc);
        finalClassifiers = [finalClassifiers, classifiers];
    end
%     save('finalClassifiers.mat','finalClassifiers');
%     [auc, prc, ~, Xfpr, Ytpr] = fusion(finalClassifiers, testdata);   
%     fprintf('\n auc = %f, prc = %f',auc,prc);
    [auc1, pred, Xfpr,Ytpr] = fusion(finalClassifiers, testdata);      
    fprintf('\n auc1 = %f',auc1);
    plot(Xfpr,Ytpr,'-ro','LineWidth',2,'MarkerSize',3);
    xlabel('False-Positive Rate');
    ylabel('True-Positive Rate');
    title(['ROC curve of (AUC = ' num2str(auc1) ' )']);
    fprintf('\n training auc = %f, unoptimized training auc = %f',mean(auc01), mean(auc02));
end



