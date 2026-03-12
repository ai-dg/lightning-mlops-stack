
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

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader as DL, TensorDataset
import lightning as L
from ydata_profiling import ProfileReport
from sklearn.preprocessing import MultiLabelBinarizer


# 1 - Detect which features had NaN and what to replace it with
# 2 - Detect which features need to be hot one encoded
# 3 - Detect invalid data 



def get_features_answers(feature: str):
	match feature:
		case "LearnCode":
			return ["Technical documentation (is generated for/by the tool or system)",
		   			"Games or coding challenges",
		   			"Colleague or on-the-job training",
		   			"Videos (not associated with specific online course or certification)",
		   			"Online Courses or Certification (includes all media types)",
		   			"Books / Physical media",
		   			"Stack Overflow or Stack Exchange",
		   			"AI CodeGen tools or AI-enabled apps",
		   			"Blogs or podcasts",
		   			"School (i.e., University, College, etc)",
		   			"Other online resources (e.g. standard search, forum, online community)",
		   			"Coding Bootcamp"]
		
		case "DevType":
			return [
				"Academic researcher",
				"AI/ML engineer",
				"Applied scientist",
				"Architect, software or solutions",
				"Cloud infrastructure engineer",
				"Cybersecurity or InfoSec professional",
				"Data engineer",
				"Data or business analyst",
				"Data scientist",
				"Database administrator or engineer",
				"Developer, AI apps or physical AI",
				"Developer, back-end",
				"Developer, desktop or enterprise applications",
				"Developer, embedded applications or devices",
				"Developer, front-end",
				"Developer, full-stack",
				"Developer, game or graphics",
				"Developer, mobile",
				"Developer, QA or test",
				"DevOps engineer or professional",
				"Engineering manager",
				"Financial analyst or engineer",
				"Founder, technology or otherwise",
				"Product manager",
				"Project manager",
				"Retired",
				"Senior executive (C-suite, VP, etc.)",
				"Student",
				"Support engineer or analyst",
				"System administrator",
				"UX, Research Ops or UI design professional",
			]
		
		case "LanguageHaveWorkedWith":
			return [
				"Ada",
				"Assembly",
				"Bash/Shell (all shells)",
				"C",
				"C#",
				"C++",
				"COBOL",
				"Dart",
				"Delphi",
				"Elixir",
				"Erlang",
				"F#",
				"Fortran",
				"GDScript",
				"Go",
				"Groovy",
				"HTML/CSS",
				"Java",
				"JavaScript",
				"Kotlin",
				"Lisp",
				"Lua",
				"MATLAB",
				"MicroPython",
				"OCaml",
				"Perl",
				"PHP",
				"PowerShell",
				"Prolog",
				"Python",
				"R",
				"Ruby",
				"Rust",
				"Scala",
				"SQL",
				"Swift",
				"TypeScript",
				"VBA",
				"Visual Basic (.Net)",
				"Zig",
				"Mojo",
				"Gleam"
			]
		
		case "DatabaseHaveWorkedWith":
			return [
				"BigQuery",
				"Cassandra",
				"Cloud Firestore",
				"Cosmos DB",
				"Databricks SQL",
				"Datomic",
				"DuckDB",
				"DynamoDB",
				"Elasticsearch",
				"Firebase Realtime Database",
				"H2",
				"IBM DB2",
				"InfluxDB",
				"MariaDB",
				"Microsoft Access",
				"Microsoft SQL Server",
				"MongoDB",
				"MySQL",
				"Neo4j",
				"Oracle",
				"PostgreSQL",
				"Redis",
				"Snowflake",
				"SQLite",
				"Supabase",
				"ClickHouse",
				"CockroachDB",
				"Amazon Redshift",
				"Pocketbase",
				"Valkey",
			]

		case "PlatformHaveWorkedWith":
			return [
				"Amazon Web Services (AWS)",
				"Ansible",
				"APT",
				"Bun",
				"Cargo",
				"Chocolatey",
				"Cloudflare",
				"Composer",
				"Datadog",
				"Digital Ocean",
				"Docker",
				"Firebase",
				"Google Cloud",
				"Gradle",
				"Heroku",
				"Homebrew",
				"IBM Cloud",
				"Kubernetes",
				"Make",
				"Maven (build tool)",
				"Microsoft Azure",
				"MSBuild",
				"Netlify",
				"New Relic",
				"Ninja",
				"npm",
				"NuGet",
				"Pacman",
				"Pip",
				"pnpm",
				"Podman",
				"Poetry",
				"Prometheus",
				"Railway",
				"Splunk",
				"Supabase",
				"Terraform",
				"Vercel",
				"Vite",
				"Webpack",
				"Yandex Cloud",
				"Yarn"
			]
		
		case "WebframeHaveWorkedWith":
			return [
				"Angular",
				"AngularJS",
				"ASP.NET",
				"ASP.NET Core",
				"Astro",
				"Blazor",
				"Deno",
				"Django",
				"Drupal",
				"Express",
				"FastAPI",
				"Fastify",
				"Flask",
				"jQuery",
				"Laravel",
				"NestJS",
				"Next.js",
				"Node.js",
				"Nuxt.js",
				"Phoenix",
				"React",
				"Ruby on Rails",
				"Spring Boot",
				"Svelte",
				"Symfony",
				"Vue.js",
				"WordPress",
				"Axum"
			]

		case "DevEnvsHaveWorkedWith":
			return [
				"Aider",
				"Android Studio",
				"Bolt",
				"Claude Code",
				"Cline and/or Roo Cursor",
				"Eclipse",
				"IntelliJ IDEA",
				"Jupyter Notebook/JupyterLab",
				"Lovable.dev",
				"Nano",
				"Neovim",
				"Notepad++",
				"PhpStorm",
				"PyCharm",
				"Rider",
				"RustRover",
				"Sublime Text",
				"Trae",
				"Vim",
				"Visual Studio",
				"Visual Studio Code",
				"VSCodium",
				"WebStorm",
				"Windsurf",
				"Xcode",
				"Zed"
			]
	return None

	
	
	

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
		
		self.clean_data()
	 

	def drop_features(self, features:list[str]):
		self.df = self.df.drop(features, axis=1)


	def replace_nan_median(self, features: list[str]):
		self.df[features] = self.df[features].fillna(self.df[features].median())

	def replace_nan_frequent(self, features: list[str]):
		self.df[features] = self.df[features].fillna(self.df[features].mode())

	def expand_features(self, to_expand: list[str]):
		for feature in to_expand:
			valid_answers = get_features_answers(feature)
			if valid_answers is None:
				raise ValueError("Features Doesnt Exists")

			other_type = feature + "Other"

			valid_answers.append(other_type)

			# Create a dictionary of your new features filled with NaN
			new_cols_dict = {feat: 0 for feat in valid_answers}

			# Convert that dictionary to a DataFrame
			#new_features_df = pd.DataFrame(new_cols_dict, index=self.df.index)

			# Join them all at once
			#self.df = pd.concat([self.df, new_features_df], axis=1)

	def hot_one_encoding(self, initial_features: list[str]):
		for i, feature in enumerate(initial_features):
			valid_answers = get_features_answers(feature)


			#Split Examples (Answers), transform NaN Values to empty list
			data = self.df[feature].str.split(';').apply(lambda x: x if isinstance(x, list) else [])

			#Use Scikit Learn to Hot One Encode feature (Add new boolean features for each possible answer)
			#NaN put 0 to every possible answer
			mlb = MultiLabelBinarizer(classes=valid_answers)

			res = mlb.fit_transform(data)

			expanded_df = pd.DataFrame(res, columns=mlb.classes_, index=self.df.index)

			valid_set = set(valid_answers)


			other_type = feature + "Other"
		
			self.df[other_type] = data.apply(
				lambda x: 1 if any(item not in valid_set for item in x) else 0
			)

			self.df = pd.concat([self.df, expanded_df], axis=1)
		self.df.to_csv("Temp.csv")
		print(self.df.shape)

	def clean_data(self):
		self.df = self.df.loc[:,self.features]

		#Clean Currency
		self.df = self.df.dropna(subset="CompTotal")
		self.df["Currency"] = self.df["Currency"].apply(erase_str)
		self.update_currency()

		#Replace nan with median value 
		nan_to_replace_median = ["WorkExp", "YearsCode"]
		self.replace_nan_median(nan_to_replace_median)

		#Replace nan with most frequent value
		#nan_to_replace_frequent = ["EdLevel"]
		#self.replace_nan_median(nan_to_replace_frequent)

		#Expand features answer
		to_expand = ["LearnCode",
					"DevType",
					"LanguageHaveWorkedWith",
					"DatabaseHaveWorkedWith",
					"PlatformHaveWorkedWith",
					"WebframeHaveWorkedWith",
					"DevEnvsHaveWorkedWith"]

		#self.expand_features(to_expand)

		#Fix Open Choices Question
		self.hot_one_encoding(to_expand)
		
		#Add Fetures for each answer of multichoice questions
		to_encode = ["MainBranch", "Age", "EdLevel", "Employement"]


	def __str__(self):
		resume = f"{self.df}"
		columns = f"{self.df.columns}"

		return resume + "\n" + columns
	
	
	def update_currency(self):

		Salary_min = 1000
		Salary_max = 999999

		currency_table  = pd.read_csv("./datasets/currecy_2025.csv")

		# Convert CompTotalEuro with its attached currency
		series_rate = currency_table.set_index("currency")['Value']
		rate = self.df["Currency"].map(series_rate)
		self.df["CompTotalEuro"] = self.df["CompTotal"] * rate
		self.df["CompTotalEuro"] = np.where(self.df["CompTotalEuro"] > Salary_max, np.nan, self.df["CompTotalEuro"])
		self.df["CompTotalEuro"] = np.where(self.df["CompTotalEuro"] < Salary_min, np.nan, self.df["CompTotalEuro"])
		self.df = self.df.dropna(subset="CompTotalEuro")

		self.df.to_csv("./datasets/result_clean.csv")
		
		# Delete the features Currency and CompTotal
		self.drop_features(["Currency", "CompTotal"])

def main():
	datapreprocess = DataLoaderClass("./datasets/survey_results_public.csv")

	#profile = ProfileReport(datapreprocess.df, title="Test")
	#profile.to_file("reports/data_analysis.html")


if __name__ == "__main__":
	main()