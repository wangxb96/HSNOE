function [auc, pred, Xfpr,Ytpr]=fusion(classifiers, data)
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
acc = mean(decisionMatrix == Y);
% deci = double(decisionMatrix); label_y = double(Y);
% [auc2, Xfpr, Ytpr] = roc_curve(deci, label_y);
end


function [auc,stack_x, stack_y] = roc_curve(deci,label_y) %%deci=wx+b, label_y, true label
	[val,ind] = sort(deci,'descend');
	roc_y = label_y(ind);
	stack_x = cumsum(roc_y == 0)/sum(roc_y == 0);
	stack_y = cumsum(roc_y == 1)/sum(roc_y == 1);
	auc = sum((stack_x(2:length(roc_y),1)-stack_x(1:length(roc_y)-1,1)).*stack_y(2:length(roc_y),1));
 
        %Comment the above lines if using perfcurve of statistics toolbox
        %[stack_x,stack_y,thre,auc]=perfcurve(label_y,deci,1);
% 	plot(stack_x,stack_y);
% 	xlabel('False Positive Rate');
% 	ylabel('True Positive Rate');
% 	title(['ROC curve of (AUC = ' num2str(auc) ' )']);
end