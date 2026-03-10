
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
    def __init__(self, path):
        super().__init__()
        try :
            self.df = pd.read_csv(path)
        except Exception as e :
            print(f"Error : {e}")
            raise RuntimeError(f"Error : {e}")
     
    def delete_features(self, features:list[str]) :
        self.df = self.df.drop(features)

    def feature_engineering(self):
        self.df = 