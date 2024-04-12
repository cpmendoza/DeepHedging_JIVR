## Hop Tech

This repository contains the deep learning models presented in 
the report: "Non-invasive wearables for remote monitoring of HbA1c
Author: Farnoosh Ghadiri, Carlos Perez Mendoza, Pierre Rosin".

- Sub-project 1: HbA1c level predection. 
- Sub-project 2: Participnats classification into two classes,
  diabetic and non-diabetic. 

## Short description

Deep learning models are contained in the `src/models/` folder. 

`regression_model.py` predicts the HbA1c level per participant based on hop
watch signals.

`classification_model.py` defines two classes diabetic and non-diabetic based
on hop watch signals for each participant.

Examples showcasing the utilization of the pipeline can be observed in the notebooks directory.
The Python script (.py file) for executing the pipeline from the terminal can be found in the pipeline directory.

## How to run

1. Python 3.8 was used as development environment.
Create a conda or Python virtual environment and install the requirements.txt (`pip install -r requirements.txt`). Alternatively, start with an empty virtual environment and install packages during execution on as-required basis.

2. Data pre-processing utilities reside in `src/data/` folder. 
An example to run the pre-processing step is included in the notebook
`data_preprocessing.ipynb`. 

3. Model functionalities reside in `src/model/` folder. 
4. The final pipeline can be executed from the terminal by using the following command in the `pipeline` folder: `python -m pipeline`.


## Directory structure

```nohighlight
├── LICENSE
├── README.md                   <- The top-level README for this project.
├── cfgs                        <- Configuration files for pre-processing step and model parameters.
│
├── data
│   ├── external                <- Hop tech data.
│   ├── interim                 <- Hop tech data integrated into a single data structure.
│   ├── processed               <- Segments data per participan (pre-processing step for model input).
│   └── results                 <- Results.
│
├── notebooks                   <- Jupyter notebooks.
│
├── pipeline                    <- .py pipeline script.
│
├── models                      <- Folder to store trained models.
│
├── src                         <- Source code for use in this project.
│   │
│   ├── data                           <- Scripts to download or generate data.
│   │   └── data_preprocessing.py      <- Script to transform data into the right format for the models.
│   │
│   ├── models                         <- Scripts to train models and then use trained models to make
│   │   │                                 predictions
│   │   ├── regression_model.py        <- Scripts to fit and make inference for the regression task.
│   │   └── classification_model.py    <- Scripts to fit and make inference for the classification task.
│   │
│   └── visualization                  <- Scripts to compute performance metrics of the models.
│   │   │
│   │   ├── regression_metrics.py      <- Scripts to compute regression model performance metrics.
│   │   └── classification_metrics.py  <- Scripts to compute classification model performance metrics.
│   │
│   └── utils.py                <- data pre-processing utility.
│ 
│ 
└── requirements.txt            <- The file for reproducing the pip-based virtual environment.
```
