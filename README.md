# Mosquito Population Estimation using LSTM

This repository contains an LSTM model designed to estimate mosquito population based on three factors: temperature, humidity, and precipitation.

## Repository Structure

### Main Files
- **main.py**: This file is used to train and evaluate different models with various hyperparameters.
- **use_best_model.py**: This file tests the best model on unseen data.
- **eda.py**: Executes exploratory data analysis (EDA) functions.
- **config.yaml**: Contains all hyperparameters and configurations.

### Data Directory
- **data**: This folder contains datasets for training and testing.

### Source Directory (src)
- **config.py**: Loads the configurations from the `config.yaml` file.
- **data_processing.py**: Processes the data before performing the train-test split.
- **model.py**: Defines the LSTM model architecture.
- **train.py**: Contains the train-test split and training functions.
- **utils.py**: Includes various plotting and summary functions for EDA.

### Training the Model
To train and evaluate different models with various hyperparameters, run:

```sh
python main.py
```

### Testing best the Model

To test the best model on unseen data, run:

```sh
python use_best_model.py
```

### Performing Exploratory Data Analysis

To run exploratory data analysis, use:

```sh
python eda.py
```

## Configuration
All hyperparameters and configurations are defined in the `config.yaml` file. Modify this file to adjust model parameters, paths, and other settings.

## Data Processing
The `data_processing.py` script processes the raw data and prepares it for training and testing.

## Model Definition
The `model.py` file contains the definition of the LSTM model used for predicting the mosquito population.

## Training
Training functions and the train-test split logic are implemented in the `train.py` file.

## Utilities
The `utils.py` file includes various utility functions for plotting and summarizing data during the exploratory data analysis phase.