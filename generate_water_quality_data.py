import pandas as pd
import numpy as np

# Load your current dataset
df = pd.read_csv("data/train.csv")


# --- SMART SYNTHETIC WATER-QUALITY GENERATION (Option B) ---

# Rainfall influences turbidity & contamination
rain = df['rainfall'].values
humidity = df['humidity'].values
temp = df['temperature'].values
density = df['population_density'].values
sentiment = df['social_sentiment'].values

# 1. pH level (slightly affected by contamination & rainfall)
df['ph_level'] = np.clip(
    7 + np.random.normal(0, 0.7, len(df)) - (rain / 300) + (sentiment / 8),
    5.0, 9.0
)

# 2. Turbidity (heavily affected by rainfall)
df['turbidity'] = np.clip(
    (rain / 10) + np.random.normal(2, 1, len(df)),
    0, 30
)

# 3. Contamination Index (0–1 scale)
df['contamination_index'] = np.clip(
    (humidity / 150) +
    (rain / 400) +
    (density / density.max() / 3) -
    (sentiment / 3) +
    np.random.normal(0.05, 0.05, len(df)),
    0, 1
)

# 4. TDS (affected by rainfall + population)
df['tds'] = np.clip(
    300 + (rain * 1.2) + (density / 50) + np.random.normal(0, 20, len(df)),
    80, 1200
)

# 5. Water temperature (slightly above ambient)
df['water_temperature'] = np.clip(
    temp + np.random.uniform(1.0, 3.0, len(df)),
    10, 40
)

# Save new dataset
df.to_csv("train_with_water_quality.csv", index=False)
print("✅ New dataset created: train_with_water_quality.csv")
