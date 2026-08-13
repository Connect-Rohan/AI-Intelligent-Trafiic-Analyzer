import pandas as pd

# Load both routes
route_a = pd.read_csv("data/route_a.csv")
route_b = pd.read_csv("data/route_b.csv")

# Combine
data = pd.concat(
    [route_a, route_b],
    ignore_index=True
)

# Convert congestion to numbers
congestion_mapping = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
    "Critical": 3
}

data["Congestion_Code"] = data[
    "Congestion_Level"
].map(congestion_mapping)

# Save ML dataset
data.to_csv(
    "data/traffic_ml_dataset.csv",
    index=False
)

print("\n================================")
print("ML DATASET READY")
print("================================")

print("\nDataset shape:")
print(data.shape)

print("\nDataset:")
print(data)

print("\nCongestion distribution:")
print(
    data["Congestion_Level"].value_counts()
)

print(
    "\nSaved to: data/traffic_ml_dataset.csv"
)