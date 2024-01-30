function [ gmean ] = metric_gmean(y, yscore)
    C = confusionmat(y,yscore); % generate confusion matrix
    true_positives = diag(C); % get the number of true positives
    actual_positives = sum(C, 2); % get the number of actual positives
    recall = true_positives ./ actual_positives; % calculate recall
    gmean = geomean(recall);
end