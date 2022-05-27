function ACO=classifierSelectionPSO(classifierList, testData, opts)
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
        c = c > 0.6; 
        c = find(c);
        
        %% CALCULATE THE ACCURACY
        decisionMatrix = ones(length(testData(:,end)), length(c));
        for i=1:length(c)
            decisionMatrix(:,i) = allPredictions(:, c(i)) ;
        end
        decisionMatrix = mode(decisionMatrix, 2);
%         error = mean(decisionMatrix ~= testData(:,end));
        [Xfpr,Ytpr,~,auc]  = perfcurve(double(testData(:,end)), double(decisionMatrix), '1','TVals','all','xCrit', 'fpr', 'yCrit', 'tpr');
        error = 1 - auc;
    end
end