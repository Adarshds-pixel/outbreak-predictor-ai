import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

CSV_PATH = r"data/epidemiology_social_dataset_1M.csv"

def create_outbreak_label(df):
    w = {
        'cases_last_week': 0.5,
        'temperature': 0.1,
        'humidity': 0.05,
        'rainfall': 0.05,
        'population_density': 0.15,
        'social_sentiment': -0.1,
        'resource_utilization': 0.15
    }

    pop_norm = (df['population_density'] - df['population_density'].min()) / \
               (df['population_density'].max() - df['population_density'].min())

    score = (
        w['cases_last_week'] * (df['cases_last_week'] / (df['cases_last_week'].max() + 1)) +
        w['temperature'] * ((df['temperature'] - 10) / 40) +
        w['humidity'] * (df['humidity'] / 100) +
        w['rainfall'] * (df['rainfall'] / 300) +
        w['population_density'] * pop_norm +
        w['social_sentiment'] * ((1 - df['social_sentiment']) / 2) +
        w['resource_utilization'] * df['resource_utilization']
    )

    score += np.random.normal(0, 0.02, size=len(df))

    labels = pd.cut(score,
                    bins=[-999, 0.33, 0.66, 999],
                    labels=['Low', 'Medium', 'High'])

    df['outbreak_risk'] = labels
    df['risk_score'] = score
    return df

def balance_dataset(df):
    # Number of samples per class (find the smallest count to equalize)
    min_count = df['outbreak_risk'].value_counts().min()

    balanced = (
        df.groupby('outbreak_risk')
        .apply(lambda x: x.sample(min_count, random_state=42))
        .reset_index(drop=True)
    )

    print("Balanced dataset counts:")
    print(balanced['outbreak_risk'].value_counts())

    return balanced

def main():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)

    print("Generating outbreak risk...")
    df = create_outbreak_label(df)

    print("Balancing dataset...")
    df = balance_dataset(df)

    print("Shuffling dataset...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if not os.path.exists("data"):
        os.makedirs("data")

    train, test = train_test_split(df, test_size=0.2,
                                   random_state=42,
                                   stratify=df["outbreak_risk"])

    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    print("✔ Balanced train/test saved!")
    print(train['outbreak_risk'].value_counts())
    print(test['outbreak_risk'].value_counts())

if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
