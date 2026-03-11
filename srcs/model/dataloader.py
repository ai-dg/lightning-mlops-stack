
#  Featrue Engineering
# squareMeters
# numberOfRooms
# hasYard
# hasPool
# floors - number of floors
# cityCode - zip code
# cityPartRange - the higher the range, the more exclusive the neighbourhood is
# numPrevOwners - number of prevoious owners
# made - year
# isNewBuilt
# hasStormProtector
# basement - basement square meters
# attic - attic square meteres
# garage - garage size -> has_garage
# hasStorageRoom
# hasGuestRoom - number of guest rooms
# price - predicted value

from sqlite3 import Cursor
from turtle import update
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader as DL, TensorDataset
import lightning as L


def erase_str(value :str):


    return value[:3]

def series_function(self):
    return

class DataLoaderClass(L.LightningDataModule):

    features = [
        "MainBranch",
        "Age",
        "EdLevel",
        "Employment",
        "WorkExp",
        "LearnCode",
        "LearnCodeAI",
        "YearsCode",
        "DevType",
        "OrgSize",
        "ICorPM",
        "RemoteWork",
        "Industry",
        "Country",
        "Currency",
        "CompTotal",
        "LanguageChoice",
        "LanguageHaveWorkedWith",
        "DatabaseChoice",
        "DatabaseHaveWorkedWith",
        "PlatformChoice",
        "PlatformHaveWorkedWith",
        "WebframeChoice",
        "WebframeHaveWorkedWith",
        "DevEnvsChoice",
        "DevEnvsHaveWorkedWith",
        "AIModelsChoice",
        "AISelect",
        "AIAgents",
    ]


    def __init__(self, path):
        super().__init__()
        try :
            self.df = pd.read_csv(path)
        except Exception as e :
            print(f"Error : {e}")
            raise RuntimeError(f"Error : {e}")
        
        self.feature_engineering()
     
    def delete_features(self, features:list[str]) :
        self.df = self.df.drop(features)

    

    def feature_engineering(self):
        self.df = self.df.loc[:,self.features]

        self.df = self.df.dropna(subset="CompTotal")

        # self.df["Currency"] = [value - 10 for key, value in self.df["Currency"]]

        # for value in self.df["Currency"]:
        #     print(value)


        self.df["Currency"] = self.df["Currency"].apply(erase_str)

        print(self.df["Currency"].unique())


    def __str__(self):
        resume = f"{self.df}"
        columns = f"{self.df.columns}"


        return resume + "\n" + columns
    
    
    def update_currency(self):

        Salary_max = 999999
        Salary_min = 1000

        currency_table  = pd.read_csv("./datasets/currecy_2025.csv")

        series_rate = currency_table.set_index("currency")['Value']
        rate = self.df["Currency"].map(series_rate)
        self.df["CompTotalEuro"] = self.df["CompTotal"] * rate
        self.df["CompTotalEuro"] = np.where(self.df["CompTotalEuro"] > Salary_max, np.nan, self.df["CompTotalEuro"])
        self.df["CompTotalEuro"] = np.where(self.df["CompTotalEuro"] < Salary_min, np.nan, self.df["CompTotalEuro"])
        self.df = self.df.dropna(subset="CompTotalEuro")
        # self.df = self.df.reset_index()
        self.df.to_csv("./datasets/result_clean.csv")
        print(self.df["CompTotalEuro"])
        print(self.df["CompTotal"])
        print(self.df.describe())
        # print(f"Sum nan country : {sum(self.df['Country'] == np.nan)}")
        # print(f" Sum nan country {self.df.loc[self.df['Country'] == np.nan].sum()}")
        self.df['Country'].isna()

        # self.df["Currency"] = currency_table.loc[self.df["Currency"]]
        # print(f"Currency : {currency_table.loc[self.df['Currency']]}")
        # self.df["CompTotal"] = self.df["CompTotal"] * currency_table.loc[:, self.df["Currency"]]


        

        

def main():
    datapreprocess = DataLoaderClass("./datasets/survey_results_public.csv")


    print(datapreprocess)

    datapreprocess.update_currency()



    # print(len(datapreprocess.df["Country"].unique()))
    # print(datapreprocess.df["Currency"].unique())
    # print(datapreprocess.df["CompTotal"])


    print(datapreprocess)



if __name__ == "__main__":
    main()