# Titanic Survival Prediction - Data Cleaning Project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# ---------------------------------------------------------
# 1. LOAD DATASET AND SUMMARIZE BASIC STATISTICS
# ---------------------------------------------------------

print("========== LOADING DATASET ==========")

df = pd.read_csv("train.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== BASIC STATISTICS ==========")
print(df.describe())


# ---------------------------------------------------------
# 2. HANDLE MISSING DATA
# ---------------------------------------------------------

print("\n========== MISSING DATA BEFORE CLEANING ==========")
print(df.isnull().sum())


# Mean imputation for Age
age_mean = df["Age"].mean()
df["Age"] = df["Age"].fillna(age_mean)

print("\nMissing Age values replaced using mean.")
print("Mean Age:", age_mean)


# Mode imputation for Embarked
embarked_mode = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(embarked_mode)

print("Missing Embarked values replaced using mode.")
print("Most common Embarked value:", embarked_mode)


# Replace missing Cabin values with Unknown
df["Cabin"] = df["Cabin"].fillna("Unknown")


print("\n========== MISSING DATA AFTER CLEANING ==========")
print(df.isnull().sum())


# ---------------------------------------------------------
# 3. ENCODE CATEGORICAL VARIABLES
# ---------------------------------------------------------

print("\n========== ENCODING CATEGORICAL VARIABLES ==========")


# Label Encoding for Sex
label_encoder = LabelEncoder()

df["Sex_Encoded"] = label_encoder.fit_transform(df["Sex"])

print("\nLabel Encoding - Sex:")
print(df[["Sex", "Sex_Encoded"]].drop_duplicates())


# One-Hot Encoding for Embarked
one_hot_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

encoded_embarked = one_hot_encoder.fit_transform(
    df[["Embarked"]]
)

encoded_columns = one_hot_encoder.get_feature_names_out(
    ["Embarked"]
)

encoded_df = pd.DataFrame(
    encoded_embarked,
    columns=encoded_columns,
    index=df.index
)

df = pd.concat([df, encoded_df], axis=1)

print("\nOne-Hot Encoding - Embarked:")
print(df[["Embarked"] + list(encoded_columns)].head())


# ---------------------------------------------------------
# 4. VISUALIZE AGE DISTRIBUTION
# ---------------------------------------------------------

print("\n========== CREATING AGE DISTRIBUTION GRAPH ==========")

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Age"],
    bins=30,
    kde=True
)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig("age_distribution.png")

plt.show()


# ---------------------------------------------------------
# 5. OUTPUT CLEANED DATASET
# ---------------------------------------------------------

print("\n========== SAVING CLEANED DATASET ==========")

df.to_csv(
    "cleaned_titanic.csv",
    index=False
)

print("Cleaned dataset saved successfully as cleaned_titanic.csv")


# ---------------------------------------------------------
# FINAL OUTPUT
# ---------------------------------------------------------

print("\n========== FINAL DATASET ==========")

print(df.head())

print("\nFinal dataset shape:", df.shape)

print("\n========== ALL TASKS COMPLETED SUCCESSFULLY ==========")