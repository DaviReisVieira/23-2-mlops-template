import pickle
import pandas as pd


class Predict:
    def __init__(self, csv_file, model_folder):
        self.csv_file = csv_file
        self.model_folder = model_folder

    def read_file(self):
        return pd.read_csv(self.csv_file)

    def drop_column(self, df):
        if "deposit" in df.columns:
            df = df.drop(labels=["deposit"], axis=1)
        return df

    def load_model(self):
        with open(f"{self.model_folder}/model.pkl", "rb") as f:
            model = pickle.load(f)

        with open(f"{self.model_folder}/ohe.pkl", "rb") as f:
            ohe = pickle.load(f)

        return model, ohe

    def predict(self, model, ohe, X):
        X_encoded = ohe.transform(X)
        y_pred = model.predict(X_encoded)
        return y_pred

    def save_file(self, df):
        df["y_pred"] = df["y_pred"].apply(lambda x: "yes" if x == 1 else "no")
        df.to_csv(self.csv_file, index=False)

    def run(self):
        df = self.read_file()
        df = self.drop_column(df)

        model, ohe = self.load_model()

        y_pred = self.predict(model, ohe, df)
        df["y_pred"] = y_pred

        self.save_file(df)

        return df


if __name__ == "__main__":
    csv_file = "data/bank_predict.csv"
    model_folder = "models"

    predictor = Predict(csv_file, model_folder)
    predictor.run()
