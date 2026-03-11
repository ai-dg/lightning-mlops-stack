
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

import pandas as pd
from torch.utils.data import DataLoader as DL, TensorDataset
import lightning as L


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

        # self.df["Currency"] = 


    def __str__(self):
        resume = f"{self.df}"
        columns = f"{self.df.columns}"


        return resume + "\n" + columns


def main():
    datapreprocess = DataLoaderClass("./datasets/survey_results_public.csv")


    print(datapreprocess)

    print(len(datapreprocess.df["Country"].unique()))
    print(datapreprocess.df["Currency"].unique())
    print(datapreprocess.df["CompTotal"])



if __name__ == "__main__":
    main()