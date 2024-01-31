clear, clc, close;
numRun = 5;
for l=1:numRun
    % Problem = {'grammatical_facial_expression01','Chen-2002','Chowdary-2006','BASEHOCK','gisette','PCMAC', 'RELATHE'}; 
    Problem = {'thyroid_sick'};
    % Problem = {'arrhythmia', 'ecoli', 'isolet', 'protein_homo', 'sick_euthyroid', 'yeast_me2', 'yeast_ml8'};
    %% MAIN LOOP
    for j = 1:length(Problem)
        p_name = Problem{j};                          
        warning('off','all');
        traindata = load(['D:\KindLab\data\HSNOE\BioData\',p_name]);
        % data = getfield(traindata, p_name);
        % if j == 1
        %     data = traindata;
        %     feat = data(:,1:end-1); 
        %     label = data(:,end);
        %     % feat = traindata.fea;
        %     % label = traindata.gnd;
        %     index = find(label == 0);
        %     label(index) = 2;
        % elseif (j > 1 && j < 4)
        %     % traindata = getfield(traindata, p_name);
        %     % data = traindata;
        %     feat = traindata.fea;
        %     label = traindata.gnd;
        % else
        %     data = [traindata.X, traindata.Y];
        %     feat = data(:,1:end-1); 
        %     label = data(:,end);
        %     index = find(label == -1);
        %     label(index) = 2;
        % end
        % Data = [feat, label];

        data = traindata.data;
        feat = data(:,1:end-1); 
        label = data(:,end);

        opts.kfold = 5;
        opts.ho = 0.1;

        %% Gaussian Mixture Model
        % GMM = jml('gmm',feat,label,opts); 
        % 
        % results.p_name = "gmm_" + p_name; 
        % results.acc = GMM.acc;
        % results.auprc = GMM.auprc;
        % results.auroc = GMM.auroc;
        % results.fscore = GMM.fscore;
        % results.gmean = GMM.gmean;  
        % saveResults(results);

        %% K Nearest Neighbor
        KNN = jml('knn',feat,label,opts); 

        results.p_name = "knn_" + p_name; 
        results.acc = KNN.acc;
        results.auprc = KNN.auprc;
        results.auroc = KNN.auroc;
        results.fscore = KNN.fscore;
        results.gmean = KNN.gmean;  
        saveResults(results);

        %% Discriminate Analysis
        % DA = jml('da',feat,label,opts);  
        % 
        % results.p_name = "da_" + p_name; 
        % results.acc = DA.acc;
        % results.auprc = DA.auprc;
        % results.auroc = DA.auroc;
        % results.fscore = DA.fscore;
        % results.gmean = DA.gmean;
        % saveResults(results);

        %% Naive Bayes ECOC
        NB = jml('nb',feat,label,opts);  

        results.p_name = "nb_" + p_name; 
        results.acc = NB.acc;
        results.auprc = NB.auprc;
        results.auroc = NB.auroc;
        results.fscore = NB.fscore;
        results.gmean = NB.gmean;
        saveResults(results);

        %% Multi Class Support Vector Machine ECOC
        MSVM = jml('msvm',feat,label,opts);  

        results.p_name = "msvm_" + p_name; 
        results.acc = MSVM.acc;
        results.auprc = MSVM.auprc;
        results.auroc = MSVM.auroc;
        results.fscore = MSVM.fscore;
        results.gmean = MSVM.gmean;
        saveResults(results);

        %% Support Vector Machine
        SVM = jml('svm',feat,label,opts);  

        results.p_name = "svm_" + p_name; 
        results.acc = SVM.acc;
        results.auprc = SVM.auprc;
        results.auroc = SVM.auroc;
        results.fscore = SVM.fscore;
        results.gmean = SVM.gmean;
        saveResults(results);

        %% Decision Tree
        DT = jml('dt',feat,label,opts);  

        results.p_name = "dt_" + p_name; 
        results.acc = DT.acc;
        results.auprc = DT.auprc;
        results.auroc = DT.auroc;
        results.fscore = DT.fscore;
        results.gmean = DT.gmean;
        saveResults(results);

        %% Random Forest
        RF = jml('rf',feat,label,opts);  

        results.p_name = "rf_" + p_name; 
        results.acc = RF.acc;
        results.auprc = RF.auprc;
        results.auroc = RF.auroc;
        results.fscore = RF.fscore;
        results.gmean = RF.gmean;
        saveResults(results);

        %% Ensemble Tree
        ET = jml('et',feat,label,opts);  

        results.p_name = "et_" + p_name; 
        results.acc = ET.acc;
        results.auprc = ET.auprc;
        results.auroc = ET.auroc;
        results.fscore = ET.fscore;
        results.gmean = ET.gmean;
        saveResults(results);
    end
end
