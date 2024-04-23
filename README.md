## Enhancing deep hedging of options with implied volatility surface feedback information

This repository contains the deep hedging environment used in our paper François et al. (2024), where we develop a dynamic hedging scheme for equity options that integrates information from a set of risk factors characterizing the implied volatility surface dynamics. This is achieved through a deep policy gradient-type reinforcement learning (RL) algorithm. The repository consists of two main components:

- Component 1: Environment generation based on the data-driven simulator JIVR introduced by François et al. (2023).
- Component 2: Implementation of the RL agent to hedge European options.

## Short description

1. The environment simulators are contained in the `src/features/` folder.

    - `nig_simulation.py` simulates NIG random vectors based on the joint distribution of the JIVR random component. This simulation considers the estimation performed using real market data, as outlined in François et al. (2022).
    - `jivr_simulation.py` simulates the JIVR environment, including underlying stock returns, volatility, and risk factors that characterize the implied volatility surface dynamics.

2. Deep RL model is contained in the `src/models/` folder. 

    - `deep_rl_agent.py` contains a class that trains and assesses the performance of RL agents based on the non-standard RNN-FNN architecture outlined in our paper François et al. (2024).

Examples showcasing the utilization of the pipeline can be observed in the notebooks directory.
The Python script (.py file) for executing the pipeline from the terminal can be found in the pipeline directory.

## How to run

1. Environment setup
Python 3.9.6 was used as development environment.

    - Clone the project repository:

```nohighlight
git clone https://github.com/OctavioPM/DeepHedging_JIVR.git
cd DeepHedging_JIVR
```

- Create and activate a virtual environment:

```nohighlight
git clone https://github.com/OctavioPM/DeepHedging_JIVR.git
cd DeepHedging_JIVR
```

- Install the requirements using `pip`

```nohighlight
pip install -r requirements.txt
```

Alternatively, start with an empty virtual environment and install packages during execution on as-required basis.

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
