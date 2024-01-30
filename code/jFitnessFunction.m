% Fitness Function KNN (9/12/2020)

function [cost, Acc, AUC, AUPRC, fscore] = jFitnessFunction(feat,label,X,num)
% Default of [alpha; beta]
ws = [0.9; 0.1];

% Check if any feature exist
if sum(X == 1) == 0
  cost = 1;
else
  % Error rate
  [error, Acc, AUC, AUPRC, fscore] = jwrapper_KNN(feat(:,X == 1),label,num);
  % Number of selected features
  num_feat = sum(X == 1);
  % Total number of features
  max_feat = length(X); 
  % Set alpha & beta
  alpha    = ws(1); 
  beta     = ws(2);
  % Cost function 
  cost     = alpha * error + beta * (num_feat / max_feat); 
%   cost = error;
end
end


%---Call Functions-----------------------------------------------------
function [error, Acc1, AUC1, AUPRC1, fscore1] = jwrapper_KNN(sFeat,label,num)
data = [sFeat,label];
k = 5;
Md = cvpartition(data(:,end),'KFold',5);
for i = 1 : 5
% Define training & validation sets
testIdx = Md.test(i);
xtrain   = sFeat(~testIdx,:); ytrain  = label(~testIdx);
xvalid   = sFeat(testIdx,:);  yvalid  = label(testIdx);
% Training model
if num == 1
   My_Model = fitcknn(xtrain,ytrain,'NumNeighbors',k); % KNN
elseif num == 2
   My_Model = fitctree(xtrain,ytrain); % DT
elseif num == 3
   My_Model = fitcdiscr(xtrain,ytrain, 'discrimtype','diaglinear'); % DISCR
elseif num == 4
   My_Model = fitcnb(xtrain,ytrain, 'distribution', 'kernel'); % NB
elseif num == 5
   My_Model = trainNN(xtrain,ytrain); % ANN
elseif num == 6      
   My_Model = fitcensemble(xtrain,ytrain,'Method','bag'); % RF
end
% Index = 1;
% Model = trainClassifiers(xtrain,ytrain, params);
% for temp = 1:length(Model)
%     classifiers{Index} = Model{1,temp};
%     Index = Index + 1;
% end
% Prediction
if num == 5
   pred  = getNNPredict(My_Model,xvalid);
else
   pred     = predict(My_Model,xvalid);
end
% Acc(i) = fusion(classifiers,[xvalid,yvalid]);
% Accuracy
Acc(i)   = sum(pred == yvalid) / length(yvalid);
% [~,~,~,Auc(i)] = perfcurve(yvalid,pred,'1'); 
% Acc(i) = metric_accuracy(yvalid,pred);
AUC(i) = metric_auroc(yvalid,pred);
AUPRC(i) = metric_auprc(yvalid,pred);
fscore(i) = metric_fscore(yvalid,pred);
end
% Error rate
error    = 1 - mean(AUC); 
Acc1 = mean(Acc);
AUC1 = mean(AUC);
AUPRC1 = mean(AUPRC);
fscore1 = mean(fscore);
%fprintf('\n Acc: %g, AUC: %g, AUPRC: %g, F1: %g %%',100 * mean(Acc), 100 * mean(AUC), 100 * mean(AUPRC), 100 * mean(fscore));
end












