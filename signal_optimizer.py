import pandas as pd

# ==========================================
# LOAD TRAFFIC DATA
# ==========================================

data = pd.read_csv(
    "data/traffic_ml_dataset.csv"
)


# ==========================================
# GET LATEST DATA FOR EACH ROUTE
# ==========================================

latest_routes = (
    data.sort_values("Time_Seconds")
    .groupby("Route")
    .tail(1)
    .copy()
)


# ==========================================
# CALCULATE TRAFFIC DEMAND
# ==========================================

total_vehicles = latest_routes[
    "Total_Vehicles"
].sum()

latest_routes["Traffic_Share"] = (
    latest_routes["Total_Vehicles"]
    / total_vehicles
)


# ==========================================
# TOTAL SIGNAL CYCLE
# ==========================================

TOTAL_SIGNAL_TIME = 90


# ==========================================
# ALLOCATE GREEN TIME
# ==========================================

latest_routes["Recommended_Green_Time"] = (
    latest_routes["Traffic_Share"]
    * TOTAL_SIGNAL_TIME
)


# ==========================================
# ROUND VALUES
# ==========================================

latest_routes[
    "Recommended_Green_Time"
] = latest_routes[
    "Recommended_Green_Time"
].round().astype(int)


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n========================================")
print("INTELLIGENT SIGNAL OPTIMIZATION")
print("========================================")

for _, row in latest_routes.iterrows():

    print(
        f"\n{row['Route']}"
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
        f"Traffic Share: "
        f"{row['Traffic_Share'] * 100:.1f}%"
    )

    print(
        f"Recommended Green Time: "
        f"{row['Recommended_Green_Time']} seconds"
    )


print("\n========================================")
print(
    f"Total Signal Cycle: "
    f"{TOTAL_SIGNAL_TIME} seconds"
)
print("========================================")