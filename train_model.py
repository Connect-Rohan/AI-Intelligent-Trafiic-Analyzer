import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv(
    "data/traffic_ml_dataset.csv"
)

print("\n================================")
print("DATASET LOADED")
print("================================")

print(data)

print("\nDataset Shape:")
print(data.shape)


# ==========================================
# 2. CHECK TARGET DISTRIBUTION
# ==========================================

print("\n================================")
print("CONGESTION DISTRIBUTION")
print("================================")

print(
    data["Congestion_Level"].value_counts()
)


# ==========================================
# 3. SELECT FEATURES
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


print("\n================================")
print("FEATURES")
print("================================")

print(features)


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

# Check whether every class has at least
# two samples.

class_counts = y.value_counts()

can_stratify = (
    len(class_counts) > 1
    and class_counts.min() >= 2
)

if can_stratify:

    print("\nUsing stratified train/test split.")

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )

else:

    print(
        "\nNot enough samples for stratification."
    )

    print(
        "Using normal train/test split."
    )

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42
    )


print("\n================================")
print("TRAIN / TEST SPLIT")
print("================================")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ==========================================
# 5. CREATE MODEL
# ==========================================

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42
)


# ==========================================
# 6. TRAIN
# ==========================================

print("\n================================")
print("MODEL TRAINING")
print("================================")

model.fit(
    X_train,
    y_train
)

print(
    "Random Forest training completed!"
)

joblib.dump(
    model,
    "traffic_model.pkl"
)

print("Model saved successfully!")
print("Saved as: traffic_model.pkl")

# ==========================================
# 7. PREDICTION
# ==========================================

y_pred = model.predict(
    X_test
)


print("\n================================")
print("PREDICTIONS")
print("================================")

print("Actual values:")

print(
    y_test.values
)

print("\nPredicted values:")

print(
    y_pred
)


# ==========================================
# 8. ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

print("\n================================")
print("CONFUSION MATRIX")
print("================================")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# 11. FEATURE IMPORTANCE
# ==========================================

print("\n================================")
print("FEATURE IMPORTANCE")
print("================================")

importance = pd.DataFrame({

    "Feature": features,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print(
    importance
)