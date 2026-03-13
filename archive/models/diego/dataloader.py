import pandas as pd
import torch
import sys

class DataLoaderCsv:

	def __init__(self, path):
		self.df = pd.read_csv(path)

		if self.ft_has_raw_features():
			self.ft_create_feature_eng()
			self.ft_erase_features([
				"Name",
				"Fare",
				"Sex",
				"Age",
				"PassengerId",
				"Ticket",
				"Cabin",
				"Embarked",
			])

		print(self.df)
		print(self.df.columns)

	def ft_has_raw_features(self):
		required_cols = ["Name", "Sex", "SibSp", "Parch"]
		return all(col in self.df.columns for col in required_cols)

	def ft_erase_features(self, labels: list[str] = []):
		try:
			self.df.drop(columns=labels, inplace=True, errors="ignore")
		except (ValueError, KeyError) as e:
			print(f"Error: {e}")
			sys.exit(1)

	def ft_create_feature_eng(self):
		self.df["Family_size"] = self.df["SibSp"] + self.df["Parch"] + 1
		self.df["is_alone"] = ((self.df["SibSp"] == 0) & (self.df["Parch"] == 0)).astype(int)
		self.df["family_group"] = (self.df["Family_size"] > 1).astype(int)
		self.df["large_family"] = (self.df["Family_size"] >= 4).astype(int)
		self.df["is_female"] = (self.df["Sex"] == "female").astype(int)
		self.df["is_master"] = self.df["Name"].str.contains("Master", case=False, na=False).astype(int)
		
	def ft_get_data_base_torch(self):

		self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

		df_half = self.df.sample(frac=0.5, random_state=42)
		df_second_half = self.df.drop(df_half.index)

		df_half = df_half.reset_index(drop=True)
		df_second_half = df_second_half.reset_index(drop=True)

		# df_half.to_csv("./datasets/test/half.csv", index=False)
		df_second_half.to_csv("./datasets/test/second_half.csv", index=False)

		y = df_half["Survived"]
		X = df_half.drop(columns="Survived")

		X = torch.tensor(X.values, dtype=torch.float32)
		y = torch.tensor(y.values, dtype=torch.long)

		n = len(X)
		train_size = int(0.8 * n)

		X_train = X[:train_size]
		X_val = X[train_size:]

		y_train = y[:train_size]
		y_val = y[train_size:]

		return X_train, X_val, y_train, y_val
	
	def ft_get_data_from_file(self):
		
		y = self.df["Survived"]
		X = self.df.drop(columns="Survived")

		X = torch.tensor(X.values, dtype=torch.float32)
		y = torch.tensor(y.values, dtype=torch.long)

		return X, y
	
	def ft_get_data_from_file_titanic(self, path_submission):
		file_y = pd.read_csv(path_submission)
		y = file_y["Survived"]
		X = self.df

		X = torch.tensor(X.values, dtype=torch.float32)
		y = torch.tensor(y.values, dtype=torch.long)

		return X, y
		

		

def main():
	loader = DataLoaderCsv("../../datasets/titanic/train.csv")



if __name__ == "__main__":
	main()