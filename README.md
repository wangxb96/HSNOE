# HSNOE
Code for “Evolving Pathway Activation from Cancer Gene Expression Data using Nature-inspired Ensemble Optimization”

## Framework
![model](https://github.com/wangxb96/HSNOE/blob/main/figures/model.png)

## Folders
- **ComparisonMethods**: The baselines for comparison, including nature-inspired methods and machine learning methods.
- **Data**: The data for the experiment.

## Instructions
### 1. Main Code
- HSNOE.m
### 2. Hybrid Sampling
- HybridSampling.m
  - NCL.m
  - SMOTE.m
    - Populate.m
### 3. Nature-inspired Feature Selection
- jAntColonyOptimization.m
### 4. Nature-inspired Ensemble Learning
- 4.1. **Subspace Generation**: generateClustersV2.m
- 4.2. **Balancing Clusters**: balanceClusters.m
- 4.3. **Ensemble Training**: trainClassifiers.m 
- 4.4. **ACO Optimization**: classifierSelectionACO.m
  - ACOptimizer.m
  - acoPredict.m
 ### 5. Model Fusion
 - fusion.m
 ### 6. ANN 
 - trainNN.m
   - prepareTarget.m
 - getNNPredict.m
 ### 7. Metrics
 - metric_accuracy.m
 - metric_auprc.m
 - metric_auroc.m
 - metric_fscore.m
 - metric_gmean.m
 ### 8. Fitness Function
 - jFitnessFunction.m
 - jknn.m
 ### 9. Save Results
 - saveResults.m
## Contact
wangxb19 at mails.jlu.edu.cn
