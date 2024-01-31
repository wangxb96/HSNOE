function rs = saveResults(results)
  fid = [];
    if (exist([pwd filesep 'results.csv'], 'file') == 0)
        fid = fopen([pwd filesep 'results.csv'], 'w');
        fprintf(fid, '%s, %s, %s, %s, %s, %s\n', ...
            'Data Set', 'Accuracy', 'AUPRC', 'AUROC', 'Fscore', 'Gmean');
    elseif (exist([pwd filesep 'results.csv'], 'file') == 2)
        fid = fopen([pwd filesep 'results.csv'], 'a');
    end
    fprintf(fid, '%s, ', results.p_name);
    fprintf(fid, '%f, %f, %f, %f, %f\n', ...
          results.acc, results.auprc, results.auroc, results.fscore, results.gmean);
    fclose(fid);
end



