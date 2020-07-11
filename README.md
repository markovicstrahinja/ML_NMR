[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markovicstrahinja/ML_NMR/blob/master/Model%20Inference.ipynb)

# NMR viscosity

The python script for data-driven prediction of viscosity of liquid hydrocarbons, based on their NMR spin-spin (T2) relaxation.

# Dataset

Dataset is available in two formats:
- Excel: download full dataset
- CSV: download [train](https://raw.githubusercontent.com/markovicstrahinja/ML_NMR/master/data/train_data.csv) (data/train_data.csv) and [test](https://raw.githubusercontent.com/markovicstrahinja/ML_NMR/master/data/test_data.csv) data (data/train_data.csv)

For the public dataset, we provide 282 samples with the following features:
 - T2LM (ms): T2-relaxation geometric mean
 - T (K): temperature of the oil sample 
 - TE (ms): echo-spacing
 - \[**Target**\] Eta (cP): viscosity
 
# Get started

To start using trained models, press Colab badge at README header or here: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markovicstrahinja/ML_NMR/blob/master/Model%20Inference.ipynb)

To train model from scratch:

1) Clone the repository: 
    ```shell script 
    git clone https://github.com/markovicstrahinja/ML_NMR.git 
    ```
2) Select one of the base model from "models/" folder or use your own Sklearn model. To use your model, 
you have save your *model* via *joblib* library into *model_file_name*: 
    ```python
    import joblib
    joblib.dump(model, model_file_name)
    ```
3) Create JSON file params.json with path to the base model (see svr_params.json or gbrt_params.json for example):
    ```json
    { 
       "base-model": "model_file_name" 
    }
    ```
4) Run training script and save new trained model into data/new_trained_model.joblib:
    ```shell script
    python train.py -lfv -m models/new_trained_model.joblib --params-json params.json    
    ```
   where input arguments denote by:
    * l –– build prediction for log(Eta) instead if Eta;
    * f -- use feature engineering;
    * v -- print output (verbose);
    * m -- output filename for trained model;
    * params-json -- path to file with model location.
5) Evaluate new trained model on test data (argument definition are the same):
    ```shell script
    python test.py -lfv -m models/new_trained_model.joblib
    ```

## Format of Grid-search

To run grid-search over model's params, you should make two steps:
1) Append model's parameters grid to params.json (see svr_params.json or gbrt_params.json for example), e.g.:
    ```json
    { 
       "base-model": "data/GradientBoostingRegressor_weights.joblib",
       "grid-search": {
           "n_estimators": [100, 200, 500],
           "max_depth": [1, 2, 5],
           "learning_rate": [1e-5, 1e-3, 1e-1]
       }
    }
    ```
2) Use grid-search mode to training script (append flag **-g** to the script):
    ```shell script
    python train.py -glfv -m models/new_trained_model.joblib --params-json params.json
    ```

If you have any questions about training procedure, please feel free to start new issue or 
directly write to [strahinja.markovic@skolkovotech.ru](mailto:strahinja.markovic@skolkovotech.ru). 

# Scoreboard

| Model | RMSE (mPa·s) | MAE (mPa·s) | MSLE | MAPE (%) | Adj R^2 (%) |
|-------|:------------:|:-----------:|:----:|:--------:|:-----------:|
|Straley	|	28,910	|	9638	|	1.014	|	54	|	0.58|
|Sandor	|	25,012	|	8066	|	0.831	|	50	|	0.83|
|Nicot	|	21,489	|	7306	|	0.712	|	67	|	0.46|
|Cheng	|	21,371	|	7085	|	0.990	|	95	|	0.59|
|DT	|	48,838	|	11,179	|	0.416	|	59	|	-2.12|
|MLR	|	30,282	|	10,858	|	3.443	|	282	|	-0.20|
|KNN	|	19,814	|	6745	|	0.261	|	50	|	0.48|
|RF	|	17,812	|	5932	|	0.226	|	54	|	0.58|
|SVR	|	22,538	|	6573	|	0.481	|	71	|	0.33|
|GBRT	|	11,514	|	4051	|	0.176	|	35	|	0.82|
|DT-FE	|	45,077	|	8215	|	0.289	|	55	|	-1.67|
|MLR-FE	|	13,572	|	4653	|	0.283	|	52	|	0.75|
|KNN-FE	|	14,559	|	4182	|	0.210	|	39	|	0.72|
|RF-FE	|	13,453	|	4674	|	0.180	|	39	|	0.76|
|SVR-FE	|	**5418**	|	**1671**	|	0.257	|	50	|	**0.96**|
|GBRT-FE	|	7140	|	2742	|	**0.123**	|	**27**	|	0.93|

Please write to [strahinja.markovic@skolkovotech.ru](mailto:strahinja.markovic@skolkovotech.ru) for posting algorithms to the scoreboard.

TODO: SM please check the numbers for SVR, GBRT, SVR-FE and GBRT-FE  

# Citation
If you use this dataset or code please cite:

TODO: LINK TO THE ARTICLE BOTH IN APA AND BIBTEX FORMAT