clear, clc, close;
numRun = 10;
for l=1:numRun

   Problem = {'ecoli', 'sick_euthyroid', 'yeast_me2', 'arrhythmia', 'yeast_ml8'}; 

    %% MAIN LOOP
    for j = 1:length(Problem)
        p_name = Problem{j};
        results.p_name = p_name;                           
        warning('off','all');
        traindata = load(['C:\Users\wangxb\Desktop\HSNOE\data\',p_name]); %s2302

        Data = traindata.data;
        feat = Data(:,1:end-1); 
        label = Data(:,end);
        index = find(label == -1);
        label(index) = 0;

        Data = [feat, label];

        % Data =data;
        cv = cvpartition(Data(:,end), 'holdout', 0.1);
        idxs = cv.test;
        TestData = Data(idxs,:);
        TrainData = Data(~idxs,:);

        data = HybridSampling(TrainData);
        ortest = TestData;
        batest = HybridSampling(TestData);

        %% Model SETTINGS
        params.classifiers = {'KNN', 'DT', 'DISCR', 'NB', 'ANN'};
        
        %% MAIN LOOP      
        Acc0 = jknn(data(:,1:end-1),data(:,end));
        results.initial_acc = mean(Acc0);
        
       %% basic settings of ACO algorithm
        num = 5;
        opts.N  = 100;     % number of solutions 100
        opts.T  = 50;    % maximum number of iterations
        opts.num = num; 
        
       %% Feature selection in each cluster by ACO algorithm
        fun = @jFitnessFunction; %set the objective function 
        ACO = jAntColonyOptimization(fun,data(:,1:end-1),data(:,end),opts);
        % Training results of nature-inspired optimization
        results.fs_training_acc = ACO.Acc;
        results.fs_training_auc = ACO.AUC;
        results.fs_training_auprc = ACO.AUPRC;
        results.fs_training_fscore = ACO.fscore;
    
        % Test results of nature-inspired optimization
        sf = ACO.sf;
        data = [data(:,sf), data(:,end)];       
        ortest = [ortest(:,sf),ortest(:,end)];
        batest = [batest(:,sf),batest(:,end)];

%         testdata = [testdata(:,sf),testdata(:,end)];     
    
%         fstrain = [data(:,sf), data(:,end)];
%         fsortest = [testdata(:,sf),testdata(:,end)];
%         fsbatest = [balancedtest(:,sf),balancedtest(:,end)];
%         save("norfstrain.mat","fstrain");
%         save("norfsortest.mat","fsortest");
%         save("norfsbatest.mat","fsbatest");
    
    %     test = [test(:,sf),test(:,end)];  
    
    
%         batest = fsbatest;
        fs_model = trainNN(data(:,1:end-1), data(:,end)); % ANN
        pred  = getNNPredict(fs_model,batest(:,1:end-1));
        results.fs_test_acc = metric_accuracy(batest(:,end), pred);
        results.fs_test_auc = metric_auroc(batest(:,end), pred);
        results.fs_test_auprc = metric_auprc(batest(:,end), pred);
        results.fs_test_fscore = metric_fscore(batest(:,end), pred);
        results.fs_test_gmean = metric_gmean(batest(:,end), pred);

        tdata = data;
%         tdata = normalize(tdata,"range");
%         batest = normalize(batest,"range");
        finalClassifiers = [];
        cvs = cvpartition(tdata(:,end), 'KFold', 5);
        for f = 1 : 5
            idxs = cvs.test(f);
            testData = tdata(idxs,:);
            trainData = tdata(~idxs, :);
            
            XX = trainData(:,1:end-1);
            YY = trainData(:, end);
            cv = cvpartition(YY,'holdout',0.2);
            indx = cv.test;
            trainX = XX(~indx,:);
            trainy = YY(~indx);
            valX = XX(indx,:);
            valy = YY(indx);
    
            trainX(isnan(trainX)) = -1;
            valX(isnan(valX)) = -1;
    
            validation = [valX,valy];
    
%             X = trainX;
%             y = trainy;
            allClusters = generateClustersV2([trainX, trainy]);
            [allClusters, centroids] = balanceClusters(allClusters, [trainX trainy]);
%             allClusters = generateClustersV2(trainData);
%             [allClusters, centroids] = balanceClusters(allClusters, trainData);
            for i = 1 : length(allClusters)
                balancedClusters{i} = allClusters{i};
            end
            balancedClusters{i+1} = [trainX, trainy];
            classifierIndex = 1;
            for c=balancedClusters
                X = c{1,1}(:, 1:end-1);
                y = c{1,1}(:, end);
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

            auc = zeros(1,size(pred,2));
            for i = 1 : size(pred,2)
                auc(i) = metric_auroc(validation(:,end), pred(:,i));
            end    
            for i = 1 : size(pred,2)
                if auc(i) >= mean(auc)
                   SC = [SC, i];
                end
            end
            classifierspre = classifiers;
            classifiers = classifiers(:,SC);
            ACO = classifierSelectionACO(classifiers, testData, opts);
            [auc01(f), ~, ~, ~, Acc(f), AUC(f), AUPRC(f), fscore(f), gmean(f)] = fusion(classifiers(ACO.sc), testData);
            [auc02(f), ~, ~, ~, Acc2(f), AUC2(f), AUPRC2(f), fscore2(f), gmean2(f)] = fusion(classifiers, testData);
            %disp(auc01(f)); disp(auc02(f));
            fprintf('\n training auc = %f, unoptimized training auc = %f, auc2 = %f, auprc = %f, fscore = %f, gmean = %f',mean(auc01), mean(auc02), mean(AUC2), mean(AUPRC2), mean(fscore2), mean(gmean2));
    %         classifiers = classifiers(ACO.sc);
            finalClassifiers = [finalClassifiers, classifiers(ACO.sc)];
%             [auc1, pred, Xfpr,Ytpr,testAcc(f),testAUC(f),testAUPRC(f),testfscore(f), testgmean(f)] = fusion(classifiers(ACO.sc), batest);  
            [~, ~, ~,~,testAcc1(f),testAUC1(f),testAUPRC1(f),testfscore1(f), testgmean1(f)] = fusion(classifiers, batest); 
            [~,~,~,~,testAcc2(f),testAUC2(f),testAUPRC2(f),testfscore2(f),testgmean2(f)] = fusion(classifiers(ACO.sc), ortest);
        end
        % save('finalClassifiers.mat','finalClassifiers');
        [auc1, pred, Xfpr,Ytpr,testAcc,testAUC,testAUPRC,testfscore, testgmean] = fusion(finalClassifiers, batest); 
        [auc3, pred3, Xfpr3,Ytpr3,testAcc3,testAUC3,testAUPRC3,testfscore3, testgmean3] = fusion(finalClassifiers, ortest); 
    %   % Optimization
        results.op_training_acc = mean(testAcc3);
        results.op_training_auc = mean(testAUC3);
        results.op_training_auprc = mean(testAUPRC3);
        results.op_training_fscore = mean(testfscore3);
        results.op_training_gmean = mean(testgmean3);
        % Unoptimization
        results.unop_training_acc = mean(testAcc2);
        results.unop_training_auc = mean(testAUC2);
        results.unop_training_auprc = mean(testAUPRC2);
        results.unop_training_fscore = mean(testfscore2);
        results.unop_training_gmean = mean(testgmean2);
        % Test results after optimization of ensemble
            
        results.op_test_acc = mean(testAcc);
        results.op_test_auc = mean(testAUC);
        results.op_test_auprc = mean(testAUPRC);
        results.op_test_fscore = mean(testfscore);
        results.op_test_gmean = mean(testgmean);
        fprintf('\n Without optimization: test acc = %f, test auc = %f, test auprc = %f, test f1score = %f, test gmean = %f',mean(testAcc1), mean(testAUC1),mean(testAUPRC1), mean(testfscore1), mean(testgmean));
        fprintf('\n Optimization for testdata: test acc = %f, test auc = %f, test auprc = %f, test f1score = %f, test gmean = %f',mean(testAcc2), mean(testAUC2),mean(testAUPRC2), mean(testfscore2), mean(testgmean2));

        results.fs_training_acc = results.fs_test_gmean;
        results.fs_training_auc = results.op_training_gmean;
        results.fs_training_auprc = results.unop_training_gmean;
        results.fs_training_fscore = results.op_test_gmean;
        saveResults(results);
    end
end



