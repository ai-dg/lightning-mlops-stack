import numpy as np
import pandas as pd
from torch.utils.data import DataLoader as DL, TensorDataset
import lightning as L
from ydata_profiling import ProfileReport
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder, OneHotEncoder
from category_encoders import TargetEncoder
from features_answer import get_features, get_features_answers



def erase_str(value :str):
	return value[:3]

class DataLoaderClass(L.LightningDataModule):

	def __init__(self, path):
		super().__init__()
		try :
			self.df = pd.read_csv(path)
			self.features = get_features()
		except Exception as e :
			print(f"Error : {e}")
			raise RuntimeError(f"Error : {e}")
		
		self.clean_data()

	######################################################################
	##### 						CLEAN DATA							 #####
	######################################################################
	

	def clean_data(self):
		self.df = self.df[self.features].copy()

		#Clean Currency
		self.df = self.df.dropna(subset="CompTotal")
		self.df.loc[:, "Currency"] = self.df["Currency"].apply(erase_str)
		self.update_currency()

		need_numerical_encoding = ["WorkExp", "YearsCode"]

		need_binary_encoding = [
								"LanguageChoice",
								"DatabaseChoice",
								"PlatformChoice",
								"WebframeChoice",
								"DevEnvsChoice",
								"AIModelsChoice"
								]

		#Ordinal Encode every Nominal Features by order of importance
		need_ordinal_encoding = ["EdLevel", "AISelect"]

		#One hot encode every Nominal Features with no order importance
		need_hot_encoding = [
							"MainBranch",
							"Age",
							"Employment",
							"DevType",
							"OrgSize",
							"ICorPM",
							"RemoteWork",
							"Industry",
							"AIAgents",
							"LearnCodeAI"
							]

		need_multi_label_encoding = [
									"LearnCode",
									"LanguageHaveWorkedWith",
									"DatabaseHaveWorkedWith",
									"PlatformHaveWorkedWith",
									"WebframeHaveWorkedWith",
									"DevEnvsHaveWorkedWith"
									]
		
		need_target_encoding = ["Country"]


		self.numerical_encoding(need_numerical_encoding)
		self.binary_encoding(need_binary_encoding)
		self.ordinal_encoding(need_ordinal_encoding)
		self.one_hot_encoding(need_hot_encoding)
		self.multi_label_encoding(need_multi_label_encoding)
		self.target_encoding(need_target_encoding)


	def update_currency(self):

		Salary_min = 1000
		Salary_max = 999999

		# Use float64 limit as a ceiling
		FLOAT_MAX = np.finfo(np.float64).max

		currency_table  = pd.read_csv("./datasets/currecy_2025.csv")

		# Convert CompTotalEuro with its attached currency
		series_rate = currency_table.set_index("currency")['Value']
		rate = self.df["Currency"].map(series_rate)

		is_safe = (rate.notna()) & (rate > 0) & (self.df["CompTotal"] < (FLOAT_MAX / rate))

		#Calculate Euro Value for safe input
		self.df.loc[:, "CompTotalEuro"] = np.where(
											is_safe, 
											self.df["CompTotal"] * rate, 
											np.nan
											)

		#Spot Invalid Value and drop them
		mask = (self.df["CompTotalEuro"] >= Salary_min) & (self.df["CompTotalEuro"] <= Salary_max)
		self.df = self.df[mask].copy()

		self.df.to_csv("./datasets/result_clean.csv")
		
		# Delete the features Currency and CompTotal
		self.drop_features(["Currency", "CompTotal"])
		


	######################################################################
	##### 						UTILS							     #####
	######################################################################
	def drop_features(self, features:list[str]):
		self.df.drop(columns=features, inplace=True)


	def replace_nan_median(self, feature: str):
		median = self.df[feature].median()
		self.df[feature].fillna(median, inplace=True)

	def replace_nan_frequent(self, feature: str):
		most_frequent = self.df[feature].mode()[0]
		self.df[feature].fillna(most_frequent, inplace=True)

	######################################################################
	##### 						ENCODING							 #####
	######################################################################
	

	def numerical_encoding(self, features: list[str]):
		for feature in features:

			#For Numerical Encoding replace NaN with most median value
			self.replace_nan_median(feature)

	def ordinal_encoding(self, initial_features: list[str]):
		for feature in initial_features:

			percentage_nan = self.df[feature].isna().sum() / len(self.df[feature])
			valid_answers = get_features_answers(feature)

			#If Nan > 20% set NaN value to -1 else replace them with the most frequent value
			if (percentage_nan > 0.2):
				placeholder = feature + "_NaN"
				self.df[feature] = self.df[feature].fillna(placeholder)

				if placeholder not in valid_answers:
					valid_answers = [placeholder] + valid_answers
			else:	
				self.replace_nan_frequent(feature)

			encoder = OrdinalEncoder(categories=[valid_answers],
							handle_unknown='use_encoded_value',
							unknown_value=-1)

			self.df[feature] = encoder.fit_transform(self.df[[feature]])

	def binary_encoding(self, features: list[str]):
		for feature in features:
			#Replace NaN Value by False
			self.df[feature].fillna(0, inplace=True)

			self.df[feature] = self.df[feature].replace("Yes", 1)
			self.df[feature] = self.df[feature].replace("No", 0)
			


	def one_hot_encoding(self, initial_features: list[str]):
		for feature in initial_features:

			percentage_nan = self.df[feature].isna().mean()
			
			use_na = True if percentage_nan > 0.2 else False

			df_encoded = pd.get_dummies(
				self.df[feature],
				dummy_na=use_na,
				drop_first=True,
				dtype=int)

			new_column_names = [f"{feature}_{i}" for i in range(1, len(df_encoded.columns) + 1)]
			df_encoded.columns = new_column_names

			self.df = pd.concat([self.df, df_encoded], axis=1)
		
		self.df.drop(columns=initial_features, inplace=True)


	def multi_label_encoding(self, initial_features: list[str]):
		for i, feature in enumerate(initial_features):
			valid_answers = get_features_answers(feature)

			#Calculate the percentage of NaN for the current feature
			percentage_nan = self.df[feature].isna().sum() / len(self.df[feature])

			#If Nan > 20% create a new feature "Unknown" else it will put 0 into all expanded features
			if (percentage_nan > 0.2):
				data = self.df[feature].str.split(';').apply(lambda x: x if isinstance(x, list) else [feature + "_NaN"])
			else:
				data = self.df[feature].str.split(';').apply(lambda x: x if isinstance(x, list) else [])

			#Use Scikit Learn to Hot One Encode feature (Add new boolean features for each possible answer)
			#NaN put 0 to every possible answer
			mlb = MultiLabelBinarizer(classes=valid_answers)

			res = mlb.fit_transform(data)

			#Create genereic name for the new columns
			expanded_features_name = [f"{feature}_{i}" for i in range(1, len(valid_answers) + 1)]

			#Transform new columns into a Dataframe
			expanded_df = pd.DataFrame(res, columns=expanded_features_name, index=self.df.index)

			#Add extra invalid answers into a new column feature_Other
			valid_set = set(valid_answers)

			other_type = feature + "_Other"
		
			self.df[other_type] = data.apply(
				lambda x: 1 if any(item not in valid_set for item in x) else 0
			)

			#Add Expanded Features Dataframe to the initial dataset
			self.df = pd.concat([self.df, expanded_df], axis=1)

		#Drop Initial Features
		self.df.drop(columns=initial_features, inplace=True)

	def target_encoding(self, features: list[str]):
		for feature in features:
			encoder = TargetEncoder(cols=[feature], smoothing=10.0)
			result = encoder.fit_transform(self.df[[feature]], self.df["CompTotalEuro"])
			self.df[feature] = result[feature]	
				
	 

			

	def __str__(self):
		resume = f"{self.df}"
		columns = f"{self.df.columns}"

		return resume + "\n" + columns
	
	

def main():
	datapreprocess = DataLoaderClass("./datasets/survey_results_public.csv")
	datapreprocess.df.to_csv("Temp.csv")

	EDA = profile = ProfileReport(datapreprocess.df, title="Data (After Cleaning)")
	profile.to_file("reports/data_analysis.html")

if __name__ == "__main__":
	main()