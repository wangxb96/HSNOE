% K-nearest Neighbor (9/12/2020)
function AUC = jknn(feat,label)
% Default of k-value
k = 5;
data = [feat,label];
Model = cvpartition(data(:,end),'KFold',5);
for i = 1 : 5
% Define training & validation sets
testIdx = Model.test(i);
xtrain   = feat(~testIdx,:);  ytrain  = label(~testIdx);
xvalid   = feat(testIdx,:);   yvalid  = label(testIdx);
% Training model
My_Model = fitcknn(xtrain,ytrain,'NumNeighbors',k); 
% Prediction
pred     = predict(My_Model,xvalid);
% Accuracy
% Acc(i)      = sum(pred == yvalid) / length(yvalid);
% AUC
[X,Y,T,AUC(i)] = perfcurve(yvalid,pred,'1'); 
end
fprintf('\n AUC: %g %%',100 * mean(AUC));
end
