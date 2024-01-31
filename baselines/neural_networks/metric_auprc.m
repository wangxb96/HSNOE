function [ AUPRC ] = metric_auprc(y, yscore, in)
% AUPRC(y, yscore) Calculates area under the precision-recall curve.
% 
% INPUT
%   y       		true labels
%   yscore  		decision values (probabilities, SVM output, etc.)
%	varargin{1}		positive label; defaults to +1
%
% RETURNS
%   auroc   		area under the ROC curve
%   opt_tp  		TP rate at the optimal point on the curve
%   opt_fp  		FP rate at the optimal point on the curve

% assert(nargin >= 2)
% if nargin > 2
% 	pos_label = varargin{1};
% else
% 	pos_label = 1;
% end
pos_label = in;

% Calculate recall(X) and precision (Y)
[X,Y] = perfcurve(y, yscore,pos_label,'xCrit','reca','yCrit','prec');

% sort data (based on accending order of X)
[X, idx] = sort(X);
Y = Y(idx);

% Remove invalid data (Y)
idxNaN = isnan(Y);
X(idxNaN) = [];
Y(idxNaN) = [];

% If X doesn't start at zero, connect Y using a horizontal line
if X(1) ~= 0
    Y = [Y(1); Y];
    X = [0; X];
end

% Calculate Area Under the Precision-Recall Curve
AUPRC = trapz(X,Y); 

% Plot the Precision-Recall Curve
% plot(X,Y);
% xlabel('Recall')
% ylabel('Precision')
% title('Precision-Recall Curve (AUPRC)')
% text(0.4,0.5,['AUPRC = ' num2str(round(AUPRC,4))])
end