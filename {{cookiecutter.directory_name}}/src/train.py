import pickle
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder


class Train:
    def __init__(self, csv_file, model_folder):
        self.csv_file = csv_file
        self.model_folder = model_folder

    def read_file(self):
        return pd.read_csv(self.csv_file)

    def split_data(self, df):
        X = df.drop(labels=["deposit"], axis=1)
        y = df["deposit"]
        return X, y

    def train_model(self, X_train, y_train):
        model = LGBMClassifier()
        model.fit(X_train, y_train)
        return model

    def save_model(self, model, ohe):
        with open(f"{self.model_folder}/model.pkl", "wb") as f:
            pickle.dump(model, f)

        with open(f"{self.model_folder}/ohe.pkl", "wb") as f:
            pickle.dump(ohe, f)

    def run(self):
        df = self.read_file()
        X, y = self.split_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1912)

        cat_cols = ["job", "marital", "education", "housing"]
        ohe = make_column_transformer(
            (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
            remainder="passthrough")

        X_train_encoded = ohe.fit_transform(X_train)
        X_test_encoded = ohe.transform(X_test)

        model = self.train_model(X_train_encoded, y_train)
        self.save_model(model, ohe)

        return model, X_test_encoded, y_test


if __name__ == "__main__":
    csv_file = "data/bank_preprocessed.csv"
    model_folder = "models"

    trainer = Train(csv_file, model_folder)
    trained_model, X_test_encoded, y_test = trainer.run()
