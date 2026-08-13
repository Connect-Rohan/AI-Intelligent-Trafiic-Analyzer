import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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

class_counts = y.value_counts()

if class_counts.min() >= 2:

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )

else:

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42
    )


model = RandomForestClassifier(

    n_estimators=100,

    random_state=42
)

model.fit(
    X_train,
    y_train
)


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
# 5. SIGNAL RECOMMENDATION
# ==========================================

def signal_recommendation(congestion_code):

    if congestion_code == 0:

        return 30

    elif congestion_code == 1:

        return 45

    elif congestion_code == 2:

        return 60

    elif congestion_code == 3:

        return 90

    return 30


# ==========================================
# 6. PREDICT ROUTE
# ==========================================

def predict_route(
    route_name,
    cars,
    motorcycles,
    buses,
    trucks,
    total_vehicles,
    density
):

    input_data = pd.DataFrame({

        "Cars": [cars],

        "Motorcycles": [motorcycles],

        "Buses": [buses],

        "Trucks": [trucks],

        "Total_Vehicles": [
            total_vehicles
        ],

        "Density": [density]

    })

    prediction = model.predict(
        input_data
    )[0]

    congestion = congestion_names.get(
        prediction,
        "Unknown"
    )

    green_time = signal_recommendation(
        prediction
    )

    print("\n================================")
    print("TRAFFIC INTELLIGENCE")
    print("================================")

    print(
        f"Route: {route_name}"
    )

    print(
        f"Vehicles: {total_vehicles}"
    )

    print(
        f"Density: {density * 100:.1f}%"
    )

    print(
        f"Predicted Congestion: "
        f"{congestion}"
    )

    print(
        f"Recommended Green Time: "
        f"{green_time} seconds"
    )

    print("================================")

    return congestion, green_time


# ==========================================
# 7. TEST WITH ROUTE DATA
# ==========================================

route_a = data[
    data["Route"] == "Route A"
].iloc[-1]

route_b = data[
    data["Route"] == "Route B"
].iloc[-1]


predict_route(

    "Route A",

    route_a["Cars"],

    route_a["Motorcycles"],

    route_a["Buses"],

    route_a["Trucks"],

    route_a["Total_Vehicles"],

    route_a["Density"]
)


predict_route(

    "Route B",

    route_b["Cars"],

    route_b["Motorcycles"],

    route_b["Buses"],

    route_b["Trucks"],

    route_b["Total_Vehicles"],

    route_b["Density"]
)