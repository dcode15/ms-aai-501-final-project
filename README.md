# AAI-501 Final Project

### Authors

- Prema Mallikarjunan
- Douglas Code
- Saad Saeed

### Overview

This repository contains code for training machine learning models to predict which reviews in a dataset customers
will find most helpful, based on the review's text, age, length, and verification status.

### Environment Setup

All required packages are listed in `requirements.txt`. They can be installed with:

    pip install -r requirements.txt

### Project Layout

The original 5-core dataset and model comparison data can be found in the `data` directory. 
For the dataset containing all software reviews, see the [source website](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).

Source code can be found in the `src` directory. 
The entrypoint scripts for training and evaluating each model are as follows:
- Linear Regression: `src/LinearRegression.py`
- Random Forest Regression: `src/RandomForestRegressor.py`
- Recurrent Neural Network: `src/recurrent_neural_network.py`
- Transformer Neural Network: `src/transformer_neural_network.py`