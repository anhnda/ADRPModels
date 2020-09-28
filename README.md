# A comparison of machine learning models in adverse drug side effect prediction [1]
## Usage:

- Create and activate a python environment using anaconda:

    `conda env create -f py37env.yml`
    
    `conda activate py37env`
    
    ```pip install qpsolvers==1.0.5```



- To generate K-Fold data

    `python main.py -i`

- To run and evaluate a model:

    `python main.py -d DATA_NAME -m MODEL_NAME -f FEATURE_TYPE` 
    
    For example:
    `python main.py -d AEOLUS -m MF -f 2
    `

    Evaluation results containing AUC, AUPR and STDERR are stored in "./results" folder.


- To obtain options for DATA_NAME and MODEL_NAME and FEATURE_TYPE:

    `python main.py -h`


## Data

All input data is available in the "./data" folder:

 
## Reference
[1] Nguyen, Duc Anh, Canh Hao Nguyen, and Hiroshi Mamitsuka. "A survey on adverse drug reaction studies: data, tasks and machine learning methods." Briefings in Bioinformatics (2019).
 




