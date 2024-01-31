% Random Forest (10/12/2020)

function RF = mRFHO(feat,label,num_tree,ho)
% [Hold-out]
fold   = cvpartition(label,'HoldOut',ho); 
% Call train & test data
xtrain = feat(fold.training,:); ytrain = label(fold.training); 
xtest  = feat(fold.test,:);     ytest  = label(fold.test);
% Train model
Model = TreeBagger(num_tree,xtrain,ytrain,...
  'OOBPred','On',...
  'Method','Classification');
% Predict
pred = predict(Model,xtest); 
% Conversion 
num_test = size(pred,1);
Z        = zeros(num_test,1);
for j = 1:num_test
  Z(j,1) = str2double(pred{j,1});
end
% Accuracy
acc = sum(Z == ytest) / length(ytest);
% Confusion matrix
confmat = confusionmat(ytest,Z); 
% Store result
RF.acc = acc; 
RF.con = confmat; 

TP = confmat(1, 1);
FN = confmat(1, 2);
FP = confmat(2, 1);
TN = confmat(2, 2);
TPR = TP / (TP + FN);
FPR = FP / (FP + TN);
precision = TP / (TP + FP);
recall = TP / (TP + FN);
F1 = 2 * (precision * recall) / (precision + recall);

[~, ~, ~, RF.auprc] = perfcurve(ytest2,pred2, 1, 'xCrit', 'reca');
[~, ~, ~, RF.auroc] = perfcurve(ytest2,pred2, 1); 
RF.fscore = F1;
RF.gmean = sqrt(precision * recall);
fprintf('\n Accuracy (RF-HO): %g %%',100 * acc); 
end

