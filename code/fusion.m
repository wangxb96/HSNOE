function [auc, pred, Xfpr,Ytpr, Acc, AUC, AUPRC, fscore, gmean]=fusion(classifiers, data)
X = data(:, 1:end-1);
Y = data(:, end);
decisionMatrix = ones(length(X(:,1)), length(classifiers));
index = 1;
for i=1:length(classifiers)
    try
        if strcmp(classifiers{1,i}.name, 'RF') == 1
            decisionMatrix(:,index) = predict(classifiers{1,i}.model, X);
        elseif strcmp(classifiers{1,i}.name, 'KNN') == 1
            decisionMatrix(:,index) = predict(classifiers{1,i}.model, X);
        elseif strcmp(classifiers{1,i}.name, 'DT') == 1
            decisionMatrix(:,index) = predict(classifiers{1,i}.model, X);
        elseif strcmp(classifiers{1,i}.name, 'NB') == 1
            decisionMatrix(:,index) = predict(classifiers{1,i}.model, X);
        elseif strcmp(classifiers{1,i}.name, 'DISCR') == 1
            decisionMatrix(:,index) = predict(classifiers{1,i}.model, X);
        elseif strcmp(classifiers{1,i}.name, 'ANN') == 1
            decisionMatrix(:,index) = getNNPredict(classifiers{1,i}.model, X);
        end
        index = index + 1;
    catch ME
        disp(sprintf('fusion causing errors: %s',ME.identifier));
    end
end
pred = decisionMatrix;
decisionMatrix = mode(decisionMatrix, 2);
% [~,~,~,auc] = perfcurve(Y,decisionMatrix,'1');
[Xfpr, Ytpr,~,auc]  = perfcurve(double(Y), double(decisionMatrix), '1','TVals','all','xCrit', 'fpr', 'yCrit', 'tpr');
% acc = mean(decisionMatrix == Y);
% deci = double(decisionMatrix); label_y = double(Y);

gmean = metric_gmean(Y,decisionMatrix);
Acc = metric_accuracy(Y,decisionMatrix);
AUC = metric_auroc(Y,decisionMatrix);
AUPRC = metric_auprc(Y,decisionMatrix);
fscore  = metric_fscore(Y,decisionMatrix);
end
