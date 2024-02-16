<div align="center">
<h1>Evolving Pathway Activation from Cancer Gene Expression Data using Nature-inspired Ensemble Optimization</h1>

[**Xubin Wang**](https://github.com/wangxb96)<sup>12</sup> · **Yunhe Wang**<sup>1*</sup> · **Zhiqing Ma**<sup>3</sup> · **Ka-Chun Wong**<sup>4</sup> · **Xiangtao Li**<sup>2</sup>


<sup>1</sup>Hebei University of Technology · <sup>2</sup>Jilin University · <sup>3</sup>Northeast Normal University · <sup>4</sup>City University of Hong Kong

<sup>*</sup>corresponding author

[**PDF**](https://www.wangxubin.site/Paper/HSNOE_ESWA24.pdf) · [**Code**](https://github.com/wangxb96/HSNOE)

</div>

## Overview
Class-imbalanced biological datasets pose significant challenges in machine learning and data analysis tasks. Prior methods to handle imbalance rely on data oversampling, which increases computational costs and overfitting. While feature selection and ensemble learning are promising techniques, current applications in imbalanced contexts are limited. To address these challenges, we present a novel framework called Hybrid Sampling Nature-Inspired Optimization Ensemble (HSNOE) to enhance the identification of hidden responders in imbalanced biological datasets. Our contributions are three-fold: 1) A hybrid undersampling and oversampling technique to mitigate class-imbalance; 2) Integrate an ant colony optimization-based feature selection that identifies informative feature subsets; 3) An ensemble classifier integrating diverse models trained on optimized features to improve performance. The experiments conducted on the five biological datasets demonstrate that HSNOE exhibits more stable comprehensive performance across six evaluation metrics compared to ten benchmark methods. We also conducted a biological analysis specifically on the Pan-cancer dataset. 

## Framework
![model](https://github.com/wangxb96/HSNOE/blob/main/figures/model.png)
The framework of the proposed HSNOE model. It consists of four main phases: In Phase 1, the original data is pre-processed through random splitting into training and test sets at a ratio of 9:1. Oversampling and undersampling techniques are then applied to balance the classes. Phase 2 employes an ACO-based nature-inspired feature selection method to identify optimal feature sets. Phase 3 trains multiple classification models on the selected feature subsets from the previous phase. An ACO-based nature-inspired ensemble learning approach is utilized to select optimal model sets. Finally, in Phase 4, a plurality voting scheme fuses the predictions from the various models selected in Phase 3 to determine the final class prediction.

## Folders
- **code**: The codes designed by this study.
- **data**: The data for the experiment.

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

## Citation
```
@article{wang2024evolving,
  title={Evolving pathway activation from cancer gene expression data using nature-inspired ensemble optimization},
  author={Wang, Xubin and Wang, Yunhe and Ma, Zhiqiang and Wong, Ka-Chun and Li, Xiangtao},
  journal={Expert Systems with Applications},
  pages={123469},
  year={2024},
  publisher={Elsevier}
}
```

## Contact
wangxb19 at mails.jlu.edu.cn
