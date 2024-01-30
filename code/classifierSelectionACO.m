function ACO=classifierSelectionACO(classifierList, testData, opts)
warning('off','all');
try
    warning('off','all');
    allPredictions = acoPredict(classifierList, testData);
    %set optimization function to PSOAF
    fun = @ACOAF;
    ACO=ACOptimizer(fun, classifierList, opts);
catch exc
    disp(sprintf('problem with %s', exc.identifier));
end


%% OBJECTIVE FUNCTION
    function error=ACOAF(c)
        % BINARIZE THE CLASSIFIER SELECTION
        length1 = length(c);
        c = c > 0.4; 
        c = find(c);
        length2 = length(c);
        %% CALCULATE THE ACCURACY
        decisionMatrix = ones(length(testData(:,end)), length(c));
        for i=1:length(c)
            decisionMatrix(:,i) = allPredictions(:, c(i)) ;
        end
        decisionMatrix = mode(decisionMatrix, 2);
%         error = mean(decisionMatrix ~= testData(:,end));
        [Xfpr,Ytpr,~,auc]  = perfcurve(double(testData(:,end)), double(decisionMatrix), '1','TVals','all','xCrit', 'fpr', 'yCrit', 'tpr');
        error = 0.9*(1 - auc) + 0.1 * (length2 / length1);
    end
end