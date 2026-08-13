import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ==========================================
# 1. LOAD DATA
# ==========================================

data = pd.read_csv(
    "data/traffic_ml_dataset.csv"
)


# ==========================================
# 2. FEATURES
# ==========================================

features = [
    "Cars",
    "Motorcycles",
    "Buses",
    "Trucks",
    "Total_Vehicles",
    "Density"
]

X = data[features]

y = data["Congestion_Code"]


# ==========================================
# 3. TRAIN MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


# ==========================================
# 4. CONGESTION LABELS
# ==========================================

congestion_names = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Critical"
}


# ==========================================
# 5. GET LATEST ROUTE DATA
# ==========================================

latest_routes = (
    data
    .sort_values("Time_Seconds")
    .groupby("Route")
    .tail(1)
    .copy()
)


# ==========================================
# 6. ML PREDICTION
# ==========================================

latest_routes["Predicted_Code"] = model.predict(
    latest_routes[features]
)


latest_routes["Predicted_Congestion"] = (
    latest_routes["Predicted_Code"]
    .map(congestion_names)
)


# ==========================================
# 7. TRAFFIC SHARE
# ==========================================

total_vehicles = latest_routes[
    "Total_Vehicles"
].sum()


latest_routes["Traffic_Share"] = (
    latest_routes["Total_Vehicles"]
    / total_vehicles
)


# ==========================================
# 8. SIGNAL CYCLE
# ==========================================

TOTAL_SIGNAL_TIME = 90


latest_routes["Green_Time"] = (
    latest_routes["Traffic_Share"]
    * TOTAL_SIGNAL_TIME
)


latest_routes["Green_Time"] = (
    latest_routes["Green_Time"]
    .round()
    .astype(int)
)


# ==========================================
# 9. DISPLAY INTELLIGENCE
# ==========================================

print("\n")
print("==========================================")
print("       AI TRAFFIC INTELLIGENCE SYSTEM")
print("==========================================")


for _, row in latest_routes.iterrows():

    print("\n------------------------------------------")

    print(
        f"Route: {row['Route']}"
    )

    print(
        f"Vehicles: "
        f"{int(row['Total_Vehicles'])}"
    )

    print(
        f"Density: "
        f"{row['Density'] * 100:.1f}%"
    )

    print(
        f"Predicted Congestion: "
        f"{row['Predicted_Congestion']}"
    )

    print(
        f"Traffic Share: "
        f"{row['Traffic_Share'] * 100:.1f}%"
    )

    print(
        f"Recommended Green Time: "
        f"{row['Green_Time']} seconds"
    )


print("\n==========================================")
print(
    f"TOTAL SIGNAL CYCLE: "
    f"{TOTAL_SIGNAL_TIME} seconds"
)
print("==========================================")