# {{cookiecutter.directory_name}}

This repository contains code for a project that focuses on predicting whether a customer will make a bank deposit. The project includes data preprocessing, model training, and prediction steps. The primary goal is to provide a demonstration of using a LightGBM classifier to predict customer behavior based on various features.

## Template Repository

This repository is a template repository for the [MLOps Bank Deposit Prediction Project Template](https://github.com/DaviReisVieira/23-2-mlops-template).

## Project Structure

The project is organized into the following files:

- `src/predict.py`: This script reads the preprocessed data, loads the trained model and one-hot encodings, performs predictions, and saves the results in a CSV file.

- `src/proccess.py`: This script preprocesses the raw data, dropping unnecessary columns and converting categorical variables into numerical format.

- `src/train.py`: This script reads the preprocessed data, splits it into training and testing sets, applies one-hot encoding to categorical variables, trains a LightGBM model, and saves the trained model and one-hot encodings.

- `notebooks/analytics.ipynb`: This Jupyter notebook provides in-depth data analysis and visualizations of the preprocessed data. It explores the relationships between different features, identifies patterns, and derives insights from the data.

## How to Use

1. **Data Preprocessing**: Run `proccess.py` to preprocess the raw data. The preprocessed data will be saved as `data/bank_preprocessed.csv`.

2. **Model Training**: Run `train.py` to train the LightGBM model using the preprocessed data. The trained model and one-hot encodings will be saved in the `models` folder.

3. **Prediction**: Run `predict.py` to use the trained model and one-hot encodings to predict whether customers in the input CSV file will make a deposit. The predicted results will be saved in the input CSV file.

## Getting Started

1. Clone this repository:

   ```bash
   git clone https://github.com/insper-classroom/23-2-mlops-aps01-DaviReisVieira
   cd 23-2-mlops-aps01-DaviReisVieira
   ```

2. Install the required packages:

   ```bash
    pip install -r requirements.txt
   ```

3. Run the preprocessing script:

   ```bash
   python src/proccess.py
   ```

4. Run the training script:

   ```bash
    python src/train.py
   ```

5. Run the prediction script:

   ```bash
    python src/predict.py
   ```

## Dependencies

- pandas
- lightgbm
- scikit-learn

## Note

This project is for educational and demonstration purposes. It showcases how to preprocess data, train a machine learning model, and use it for predictions. For production use, further optimizations and considerations would be needed.
