# src/proccess.py
import pandas as pd


class Proccess:
    def __init__(self, csv_name, save_name):
        self.df = None
        self.csv_name = csv_name
        self.save_name = save_name

    def read_file(self):
        self.df = pd.read_csv(self.csv_name)
        return self.df

    def drop_columns(self):
        self.df = self.df.drop(labels=["default", "contact", "day", "month",
                               "pdays", "previous", "loan", "poutcome", "poutcome"], axis=1)
        return self.df

    def convert_categorical(self):
        dep_mapping = {"yes": 1, "no": 0}
        self.df["deposit"] = self.df["deposit"].astype(
            "category").map(dep_mapping)
        return self.df

    def save_file(self):
        self.df.to_csv(self.save_name, index=False)
        return self.df

    def run(self):
        self.read_file()
        self.drop_columns()
        self.convert_categorical()
        self.save_file()

        return self.df


if __name__ == "__main__":
    pre = Proccess("data/bank.csv", "data/bank_preprocessed.csv")
    pre.run()
