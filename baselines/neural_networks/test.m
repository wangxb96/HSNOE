clear, clc, close;
numRun = 5;
for l=1:numRun
    Problem = {'thyroid_sick'};
    %% MAIN LOOP
    for j = 1:length(Problem)
        p_name = Problem{j};                          
        warning('off','all');
        traindata = load(['D:\KindLab\data\HSNOE\BioData\',p_name]);
        % Data = getfield(traindata, p_name);
        % if j == 0
        %     data = traindata;
        %     feat = data(:,1:end-1); 
        %     label = data(:,end);
        %     % feat = traindata.fea;
        %     % label = traindata.gnd;
        %     index = find(label == 0);
        %     label(index) = 2;
        % elseif j < 1
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
        index = find(label == -1);
        label(index) = 2;
        data = [feat, label];

        %% Basic settings of Neural Network
        opts.tf        = 1;
        opts.ho        = 0.1;
        opts.H         = 10;
        opts.Maxepochs = 50;
        opts.kfold = 5;

        %% Feed Forward Neural Network (FFNN) with hold-out validation
        % Perform neural network 
        % FFNN = jnn('ffnn',feat,label,opts); 
        % 
        % results.p_name = "ffnn_" + p_name; 
        % results.acc = FFNN.acc;
        % results.auprc = FFNN.auprc;
        % results.auroc = FFNN.auroc;
        % results.fscore = FFNN.fscore;
        % results.gmean = FFNN.gmean;  
        % saveResults(results);

        %% Neural Network (NN) with hold-out validation
        NN = jnn('nn',feat,label,opts); 

        results.p_name = "nn_" + p_name; 
        results.acc = NN.acc;
        results.auprc = NN.auprc;
        results.auroc = NN.auroc;
        results.fscore = NN.fscore;
        results.gmean = NN.gmean;  
        saveResults(results);

        %% Neural Network (NN) with hold-out validation
        % CFNN = jnn('cfnn',feat,label,opts); 
        % 
        % results.p_name = "cfnn_" + p_name; 
        % results.acc = CFNN.acc;
        % results.auprc = CFNN.auprc;
        % results.auroc = CFNN.auroc;
        % results.fscore = CFNN.fscore;
        % results.gmean = CFNN.gmean;  
        % saveResults(results);

        %% Recurrent Neural Network (RNN) with hold-out validation
        % RNN = jnn('rnn',feat,label,opts); 
        % 
        % results.p_name = "rnn_" + p_name; 
        % results.acc = RNN.acc;
        % results.auprc = RNN.auprc;
        % results.auroc = RNN.auroc;
        % results.fscore = RNN.fscore;
        % results.gmean = RNN.gmean;
        % saveResults(results);

        %% Generalized Regression Neural Network (GRNN) with hold-out validation
        GRNN = jnn('grnn',feat,label,opts); 

        results.p_name = "grnn_" + p_name; 
        results.acc = GRNN.acc;
        results.auprc = GRNN.auprc;
        results.auroc = GRNN.auroc;
        results.fscore = GRNN.fscore;
        results.gmean = GRNN.gmean;
        saveResults(results);

        %% Probabilistic Neural Network (PNN) with hold-out validation
        PNN = jnn('pnn',feat,label,opts); 

        results.p_name = "pnn_" + p_name; 
        results.acc = PNN.acc;
        results.auprc = PNN.auprc;
        results.auroc = PNN.auroc;
        results.fscore = PNN.fscore;
        results.gmean = PNN.gmean;
        saveResults(results);
    end
end
