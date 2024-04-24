## Enhancing deep hedging of options with implied volatility surface feedback information

This repository contains the deep hedging environment used in our paper, François et al. (2024), where we develop a dynamic hedging scheme for equity options that integrates information from a set of risk factors characterizing the implied volatility surface dynamics. We employ deep policy gradient-type reinforcement learning (RL) algorithm. The repository consists of two main components:

- Component 1: Environment generation based on the data-driven simulator JIVR introduced by François et al. (2023).
- Component 2: Implementation of the RL agent to hedge European options.

## Short description

1. The environment simulators, component 1, are contained in the `src/features/` folder. 

    - `nig_simulation.py` simulates NIG random vectors based on the joint distribution of the JIVR random component. This simulation incorporates the estimation conducted using real market data, as detailed in François et al. (2022).
    - `jivr_simulation.py` simulates the JIVR environment, including underlying stock returns, volatility, and risk factors that characterize the implied volatility surface dynamics. Further theoretical simulation details can be found in François et al. (2024).

2. Deep RL model, component 2, is contained in the `src/models/` folder. 

    - `deep_rl_agent.py` contains all model functionalities through a python class that trains and assesses the performance of RL agents based on the non-standard RNN-FNN architecture outlined in our paper.

Examples showcasing the utilization of the pipeline can be observed in the notebooks directory.
The Python script (.py file) for executing the pipeline from the terminal can be found in the pipeline directory.

## How to run

1. **Prerequisities**
    - Python 3.9.6 was used as development environment.
    - The latest versions of pip

2. **Environment setup**

- Clone the project repository:

```nohighlight
git clone https://github.com/OctavioPM/DeepHedging_JIVR.git
cd DeepHedging_JIVR
```

- Create and activate a virtual environment:

```nohighlight
python -m venv venv
source venv/bin/activate
```

- Install the requirements using `pip`

```nohighlight
pip install -r requirements.txt
```

- Alternatively, start with an empty virtual environment and install packages during execution on as-required basis.

3. **Modify parameters**: The default parameters can be modified in the configuration files located in the cfgs folder:

- `config_simulation.yml`: General parameters for the simulation.

- `config_agent.yml`: Hyperparameters of the RL optimization problem.


4. **Running the script**: We provide two options to run the deep hedging JIVR pipeline:

- Option 1. The two main components of the pipeline can be executed independently following the example `deep_hedging_pipeline.ipynb` included in the `notebooks` folder. This notebook alredy outlines RL-CVaR agent performance metrics considered in Table 1 of our paper François et al. (2024).

- Option 2. The final pipeline can be executed from the terminal by using the following command in the `pipeline` folder: 

```nohighlight
cd notebooks
python -m pipeline
```

## Directory structure

```nohighlight
├── LICENSE
├── README.md                   <- The top-level README for this project.
├── cfgs                        <- Configuration files for environment simulation and RL model parameters.
│
├── data
│   ├── raw                     <- Historical estimated parameters of the JIVR model.
│   ├── interim                 <- NIG data simulation for market dynamics simulation.
│   ├── processed               <- Simulated JIVR markets dynamics.
│   └── results                 <- Deep hedging strategies (RL sgents output).
│
├── notebooks                   <- Jupyter notebook with pipeline example.
│
├── pipeline                    <- .py pipeline script.
│
├── models                      <- Folder to store trained RL agents.
│
├── src                         <- Source code for use in this project.
│   │
│   ├── data                    <- Scripts to download and generate data.
│   │   └── data_loader.py         <- Script to transform data into the right format for the models.
│   │
│   ├── features                   <- Scripts to generate market environment.
│   │   ├── jivr_simulation.py     <- Script to generate JIVR model features.
│   │   └── nig_simulation.py      <- Script to generate NIG random variables simulation.
│   │
│   ├── models                     <- Scripts to train models and then use trained models to make
│   │   │                             hedging strategies.
│   │   ├── deep_rl_agent.py       <- Script create RL agents as class objects.
│   │   └── deep_rl_training.py    <- Script to fit and make inference.
│   │
│   ├── visualization              <- Scripts to compute performance metrics of the models.
│   │   └── strategy_evalution.py  <- Scripts to compute performance metrics.
│   │
│   └── utils.py                   <- data utility for configuration files.
│ 
└── requirements.txt               <- The file for reproducing the pip-based virtual environment.
```
