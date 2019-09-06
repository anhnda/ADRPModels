# A comparison of machine learning models in adverse drug side effect prediction
Duc Anh Nguyen, Canh Hao Nguyen and Hiroshi Mamitsuka
## Usage:

Create and activate a python environment using anaconda:

`conda env create -f py37env.yml`

`conda activate py37env`



To generate K-Fold data

`python main.py -i`

To run and evaluate a model:

`python main.py -d DATA_NAME -m MODEL_NAME`


where options for DATA_NAME are "Liu" and "Aeolus" for Liu dataset [1] and Aeolus dataset [2] respectively.
Options for MODEL_NAME are "KNN", "CCA", "RF", "SVM", "RD", "GB", "LR", "MF", "NN", and "CNN"
for k-nearest neighbor, canonical correlation analysis, random forest, support vector machines, gradient boosting, logistic regression, matrix factorization, multilayer feedforward neural network, and neural fingerprint model, respectively.
"CNN" model is implemented from [3].

Evaluation results containing AUC, AUPR and STDERR are stored in "./results" folder.

All input data is available in "./data" folder. 
## Reference

[1] Liu, Mei, Yonghui Wu, Yukun Chen, Jingchun Sun, Zhongming Zhao, Xue-wen Chen, Michael Edwin Matheny, and Hua Xu. "Large-scale prediction of adverse drug reactions using chemical, biological, and phenotypic properties of drugs." Journal of the American Medical Informatics Association 19, no. e1 (2012): e28-e35.

[2] Banda JM, Evans L, Vanguri RS, Tatonetti NP, Ryan PB, Shah NH (2016) A curated and standardized adverse drug event resource to accelerate drug safety research. Scientific Data 3: 160026. https://doi.org/10.1038/sdata.2016.26

[3] Dey, Sanjoy, Heng Luo, Achille Fokoue, Jianying Hu, and Ping Zhang. "Predicting adverse drug reactions through interpretable deep learning framework." BMC bioinformatics 19, no. 21 (2018): 476.



 




