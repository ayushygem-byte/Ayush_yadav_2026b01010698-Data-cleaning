import numpy as np
import pandas as pd


def clean_titanic_data(
    filepath: str = r"C:\\Users\\ayush\\OneDrive\\Desktop\\project data analyst\\Titanic-Dataset.csv",
    output_path: str | None = None,
) -> pd.DataFrame:
    # 1. Load dataset (positional argument fixes TypeError)
    df = pd.read_csv(filepath)

    # 2. Remove exact duplicates & standardize text whitespace
    df = df.drop_duplicates().copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    # 3. Extract Title from Name (useful for analysis & imputation)
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
    rare_title_map = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Royal",
        "the Countess": "Royal",
        "Sir": "Royal",
        "Don": "Royal",
        "Jonkheer": "Royal",
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Dr": "Officer",
        "Rev": "Officer",
    }
    df["Title"] = df["Title"].replace(rare_title_map).fillna("Other")

    # 4. Handle missing values
    # Embarked: fill missing values with the mode ('S')
    if not df["Embarked"].mode().empty:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Age: impute using median age grouped by Sex and Pclass
    df["Age"] = df["Age"].fillna(
        df.groupby(["Sex", "Pclass"], observed=False)["Age"].transform("median")
    )

    # Cabin: extract first letter as Deck, fill missing with 'Unknown'
    df["Deck"] = df["Cabin"].str[0].fillna("Unknown")

    # 5. Feature engineering & data types
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Sex_encoded"] = df["Sex"].map({"male": 0, "female": 1})

    categorical_cols = ["Survived", "Pclass", "Sex", "Embarked", "Deck", "Title"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 6. Optional: save cleaned dataset
    if output_path:
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    input_file = r"C:\Users\ayush\OneDrive\Desktop\project data analyst\Titanic-Dataset.csv"
    output_file = r"C:\Users\ayush\OneDrive\Desktop\project data analyst\Titanic-Dataset-Cleaned.csv"

    cleaned_df = clean_titanic_data(
        filepath=input_file,
        output_path=output_file,
    )
    print("Remaining missing values:\n", cleaned_df.isnull().sum())
    print("\nCleaned shape:", cleaned_df.shape)