clear, clc, close;
numRun = 10;
for l=1:numRun

   Problem = {'ecoli'};

    %% MAIN LOOP
    for j = 1:length(Problem)
        p_name = Problem{j};
        results.p_name = p_name;                           
        warning('off','all');
        %% load data
        traindata = load(['C:\Users\wangxb\Desktop\HSNOE\data\,p_name]);

        Data = traindata.data;
        feat = Data(:,1:end-1); 
        label = Data(:,end);
        index = find(label == -1);
        label(index) = 0;

        Data = [feat, label];

        % Data =data;
        cv = cvpartition(Data(:,end), 'holdout', 0.1);
        idxs = cv.test;
        test = Data(idxs,:);
        train = Data(~idxs,:);

        data = HybridSampling(train);
        orriginal_test = test;
        balanced_test = HybridSampling(test);

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
        orriginal_test = [orriginal_test(:,sf),orriginal_test(:,end)];
        balanced_test = [balanced_test(:,sf),balanced_test(:,end)];


        fs_model = trainNN(data(:,1:end-1), data(:,end)); % ANN
        pred  = getNNPredict(fs_model,balanced_test(:,1:end-1));
        results.fs_test_acc = metric_accuracy(balanced_test(:,end), pred);
        results.fs_test_auc = metric_auroc(balanced_test(:,end), pred);
        results.fs_test_auprc = metric_auprc(balanced_test(:,end), pred);
        results.fs_test_fscore = metric_fscore(balanced_test(:,end), pred);
        results.fs_test_gmean = metric_gmean(balanced_test(:,end), pred);

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
    
            allClusters = generateClustersV2([trainX, trainy]);
            [allClusters, centroids] = balanceClusters(allClusters, [trainX trainy]);
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
            [~,~,~,~,testAcc1(f),testAUC1(f),testAUPRC1(f),testfscore1(f),testgmean1(f)] = fusion(classifiers, balanced_test); 
            [~,~,~,~,testAcc2(f),testAUC2(f),testAUPRC2(f),testfscore2(f),testgmean2(f)] = fusion(classifiers, orriginal_test);
            [~,~,~,~,testAcc3(f),testAUC3(f),testAUPRC3(f),testfscore3(f),testgmean3(f)] = fusion(classifiers(ACO.sc), balanced_test);
            [~,~,~,~,testAcc4(f),testAUC4(f),testAUPRC4(f),testfscore4(f),testgmean4(f)] = fusion(classifiers(ACO.sc), orriginal_test);
        end
        % save('finalClassifiers.mat','finalClassifiers');
        [~,~,~,~,testAcc5,testAUC5,testAUPRC5,testfscore5, testgmean5] = fusion(finalClassifiers, balanced_test); 
        
        %% Unoptimization
        % Balanced 
        results.unop_batest_acc = mean(testAcc1);
        results.unop_batest_auc = mean(testAUC1);
        results.unop_batest_auprc = mean(testAUPRC1);
        results.unop_batest_fscore = mean(testfscore1);
        results.unop_batest_gmean = mean(testgmean1);
        
        % Original 
        results.unop_ortest_acc = mean(testAcc2);
        results.unop_ortest_auc = mean(testAUC2);
        results.unop_ortest_auprc = mean(testAUPRC2);
        results.unop_ortest_fscore = mean(testfscore2);
        results.unop_ortest_gmean = mean(testgmean2);

        %% Optimization
        % Balanced 
        results.op_batest_acc = mean(testAcc3);
        results.op_batest_auc = mean(testAUC3);
        results.op_batest_auprc = mean(testAUPRC3);
        results.op_batest_fscore = mean(testfscore3);
        results.op_batest_gmean = mean(testgmean3);

        % Original 
        results.op_ortest_acc = mean(testAcc4);
        results.op_ortest_auc = mean(testAUC4);
        results.op_ortest_auprc = mean(testAUPRC4);
        results.op_ortest_fscore = mean(testfscore4);
        results.op_ortest_gmean = mean(testgmean4);

        % Balanced ensemble
        results.be_op_test_acc = mean(testAcc5);
        results.be_op_test_auc = mean(testAUC5);
        results.be_op_test_auprc = mean(testAUPRC5);
        results.be_op_test_fscore = mean(testfscore5);
        results.be_op_test_gmean = mean(testgmean5);

        saveResults(results);
    end
end




