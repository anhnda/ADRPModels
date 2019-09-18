# A comparison of machine learning models in adverse drug side effect prediction
Duc Anh Nguyen, Canh Hao Nguyen and Hiroshi Mamitsuka
## Usage:

- Create and activate a python environment using anaconda:

    `conda env create -f py37env.yml`
    
    `conda activate py37env`



- To generate K-Fold data

    `python main.py -i`

- To run and evaluate a model:

    `python main.py -d DATA_NAME -m MODEL_NAME -f FEATURE_TYPE` 
    
    For example:
    `python main.py -d Liu -m KNN -f 0
    `

    Evaluation results containing AUC, AUPR and STDERR are stored in "./results" folder.


- To obtain options for DATA_NAME and MODEL_NAME and FEATURE_TYPE:

    `python main.py -h`


## Data

All input data is available in "./data" folder:

- Liu_Data: [1]
- AEOLUS_Data: [2]

 
## Reference

[1] Liu, Mei, Yonghui Wu, Yukun Chen, Jingchun Sun, Zhongming Zhao, Xue-wen Chen, Michael Edwin Matheny, and Hua Xu. "Large-scale prediction of adverse drug reactions using chemical, biological, and phenotypic properties of drugs." Journal of the American Medical Informatics Association 19, no. e1 (2012): e28-e35.

[2] Banda JM, Evans L, Vanguri RS, Tatonetti NP, Ryan PB, Shah NH (2016) A curated and standardized adverse drug event resource to accelerate drug safety research. Scientific Data 3: 160026. https://doi.org/10.1038/sdata.2016.26
<!--
[3] Dey, Sanjoy, Heng Luo, Achille Fokoue, Jianying Hu, and Ping Zhang. "Predicting adverse drug reactions through interpretable deep learning framework." BMC bioinformatics 19, no. 21 (2018): 476.
-->


 




