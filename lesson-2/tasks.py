import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r'/home/az1zbekx/MAAB-BI/ml/lesson-2/datasets/housing.csv'
df = pd.read_csv(file_path)

df.head(10)

df.info()

df.describe()

df['ocean_proximity'].value_counts()

for col in df.columns:
    if df[col].isna().sum() > 0:
        print(col, df[col].isna().sum())

numerical_df = df.select_dtypes(np.number)
numerical_df

categorical_df = df.select_dtypes(include=['category'])
categorical_df

num_cols = df.select_dtypes(include=['int64','float64']).columns

for col in num_cols:
    print(f"\n--- {col} ---")
    
    mean = df[col].mean()
    median = df[col].median()
    skew = df[col].skew()
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    
    print(f"mean: {mean:.2f}, median: {median:.2f}, skew: {skew:.2f}")
    print(f"outliers: {len(outliers)}")
    
    if skew > 1:
        print("→ heavily right-skewed")
    elif skew > 0.5:
        print("→ moderately right-skewed")
    elif skew < -1:
        print("→ heavily left-skewed")
    elif skew < -0.5:
        print("→ moderately left-skewed")
    else:
        print("→ approximately normal")
    
    if len(outliers) > 0:
        print("→ outliers detected")
    else:
        print("→ no significant outliers")

def missing_report(df):
    
    data = []
    
    for column_name in df.columns:
        if df[column_name].isna().sum() > 0:
            data.append([column_name, df[column_name].isna().sum(), df[column_name].isna().sum() / len(df) * 100])
    
    return pd.DataFrame(data=data, columns=["Name", "Count", "Percentage"])

missing_report(df)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

imputer.fit(numerical_df)

data = imputer.transform(numerical_df)

numerical_df = pd.DataFrame(data=data, columns=imputer.feature_names_in_)

df = pd.concat([numerical_df, categorical_df], axis=1)
df.isna().sum()

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit(categorical_df)

one_hot_encoded = pd.DataFrame(data=enc.transform(categorical_df).toarray(), columns=enc.get_feature_names_out()).astype(int)
one_hot_encoded

from sklearn.preprocessing import StandardScaler, MinMaxScaler

cols = ["median_income", "housing_median_age", "population", "median_house_value"]

std_scale = StandardScaler().fit(df[cols])
df_std = std_scale.transform(df[cols])

minmax_scale = MinMaxScaler().fit(df[cols])
df_min_max = minmax_scale.transform(df[cols])

df_std = pd.DataFrame(df_std, columns=cols)
df_min_max = pd.DataFrame(df_min_max, columns=cols)
df[cols] = df_min_max

for col in cols:
    plt.figure(figsize=(10,5))
    
    plt.hist(df[col], bins=50, alpha=0.5, label='Original')
    
    plt.hist(df_std[col], bins=50, alpha=0.5, label='StandardScaler')
    
    plt.hist(df_min_max[col], bins=50, alpha=0.5, label='MinMaxScaler')
    
    plt.title(f"{col} distribution comparison")
    plt.legend()
    plt.show()

df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room "] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]
df

df.to_csv(file_path, index=False)