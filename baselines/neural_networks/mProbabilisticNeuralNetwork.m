% Probabilistic Neural Network 

function PNN = mProbabilisticNeuralNetwork(feat,label,opts)
% Parameters
num_spread = 0.1;
kfold      = 10; 
tf         = 2; 

if isfield(opts,'kfold'), kfold = opts.kfold; end
if isfield(opts,'ho'), ho = opts.ho; end
if isfield(opts,'tf'), tf = opts.tf; end
if isfield(opts,'nSpread'), num_spread = opts.nSpread; end 

pred2  = [];
ytest2 = [];

% [Hold-out]
if tf == 1
  fold  = cvpartition(label,'HoldOut',ho);
  K     = 1;
  
% [Cross-validation] 
elseif tf == 2
  fold  = cvpartition(label,'KFold',kfold);
  K     = kfold;
  Afold = zeros(kfold,1);
end

for i = 1:K
  % Call train & test data
  trainIdx = fold.training(i); testIdx = fold.test(i);
  xtrain   = feat(trainIdx,:); ytrain  = label(trainIdx);
  xtest    = feat(testIdx,:);  ytest   = label(testIdx); 
  % Training the model 
  net      = newpnn(xtrain',dummyvar(ytrain)',num_spread); 
  % Perform testing
  pred     = net(xtest');
  % Confusion matrix
  [~, con] = confusion(dummyvar(ytest)',pred);
  % Get accuracy for each fold
  Afold(i) = sum(diag(con)) / sum(con(:));
  % Store temporary result for each fold
  pred2    = [pred2(1:end,:), pred];
  ytest2   = [ytest2(1:end); ytest];
end
% Overall confusion matrix
[~, confmat] = confusion(dummyvar(ytest2)',pred2); 
confmat      = transpose(confmat);
% Average accuracy over k-folds
acc = mean(Afold);
% Store results 
PNN.acc  = acc;
PNN.con  = confmat;

TP = confmat(1, 1);
FN = confmat(1, 2);
FP = confmat(2, 1);
TN = confmat(2, 2);
TPR = TP / (TP + FN);
FPR = FP / (FP + TN);
precision = TP / (TP + FP);
recall = TP / (TP + FN);
F1 = 2 * (precision * recall) / (precision + recall);

pred3 = pred2';
maxIndices = zeros(size(pred3, 1), 1);
for i = 1:size(pred3, 1)
    [~, maxIndex] = max(pred3(i, :));
    maxIndices(i) = maxIndex;
end

PNN.auprc = metric_auprc(ytest2,maxIndices,mode(label));
PNN.auroc = metric_auroc(ytest2,maxIndices,mode(label));
PNN.fscore = F1;
PNN.gmean = metric_gmean(ytest2,maxIndices);

if tf == 1
  fprintf('\n Classification Accuracy (PNN-HO): %g %%',100 * acc);
elseif tf == 2
  fprintf('\n Classification Accuracy (PNN-CV): %g %%',100 * acc);
end
end

