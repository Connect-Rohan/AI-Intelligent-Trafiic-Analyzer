import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "traffic_model.pkl"

DATASET_PATH = "data/traffic_ml_dataset.csv"

METRICS_PATH = "model_metrics.pkl"


# ==========================================
# LABEL DEFINITIONS
# ==========================================

LABEL_MAP = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
    "Critical": 3
}

LABEL_NAMES = [
    "Low",
    "Moderate",
    "High",
    "Critical"
]


# ==========================================
# MODEL FEATURES
# ==========================================

FEATURE_COLUMNS = [
    "Cars",
    "Motorcycles",
    "Buses",
    "Trucks",
    "Total_Vehicles",
    "Density"
]


TARGET_COLUMN = "Congestion_Level"


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading traffic model...")

model = joblib.load(
    MODEL_PATH
)


# ==========================================
# SHOW MODEL CLASSES
# ==========================================

print(
    "\nModel classes:"
)

if hasattr(model, "classes_"):

    print(
        model.classes_
    )

else:

    print(
        "Model does not expose classes_"
    )


# ==========================================
# LOAD DATASET
# ==========================================

print(
    "\nLoading traffic dataset..."
)

data = pd.read_csv(
    DATASET_PATH
)


# ==========================================
# CHECK REQUIRED COLUMNS
# ==========================================

required_columns = (
    FEATURE_COLUMNS
    + [TARGET_COLUMN]
)


missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]


if missing_columns:

    print(
        "\nERROR: Missing required columns:"
    )

    for column in missing_columns:

        print(
            f" - {column}"
        )

    raise SystemExit


# ==========================================
# SHOW DATASET LABELS
# ==========================================

print(
    "\nDataset congestion labels:"
)

print(
    data[TARGET_COLUMN].value_counts()
)


# ==========================================
# PREPARE FEATURES
# ==========================================

X = data[
    FEATURE_COLUMNS
].copy()


# ==========================================
# PREPARE TARGET
# ==========================================

y_original = data[
    TARGET_COLUMN
].copy()


# ==========================================
# ROBUST LABEL CONVERSION
# ==========================================

def convert_label(value):

    # --------------------------------------
    # Already numeric
    # --------------------------------------

    if isinstance(
        value,
        (int, float)
    ):

        if pd.isna(value):
            return None

        return int(value)


    # --------------------------------------
    # Convert text safely
    # --------------------------------------

    value = str(
        value
    ).strip()


    # --------------------------------------
    # Text label
    # --------------------------------------

    if value in LABEL_MAP:

        return LABEL_MAP[
            value
        ]


    # --------------------------------------
    # Numeric stored as text
    # --------------------------------------

    try:

        return int(
            float(value)
        )

    except ValueError:

        return None


# Apply conversion
y_numeric = y_original.apply(
    convert_label
)


# ==========================================
# CHECK CONVERSION
# ==========================================

print(
    "\nConverted congestion labels:"
)

print(
    y_numeric.value_counts()
)


# ==========================================
# CHECK INVALID LABELS
# ==========================================

invalid_rows = y_numeric.isna()


if invalid_rows.any():

    print(
        "\nWARNING: Some labels could not be converted:"
    )

    print(
        y_original[
            invalid_rows
        ].unique()
    )


# ==========================================
# REMOVE INVALID ROWS
# ==========================================

valid_rows = (
    ~y_numeric.isna()
)


X = X.loc[
    valid_rows
].copy()


y_numeric = y_numeric.loc[
    valid_rows
].astype(int)


# ==========================================
# SAFETY CHECK
# ==========================================

if len(X) == 0:

    print(
        "\nERROR: No valid rows remain."
    )

    print(
        "Check the Congestion_Level values "
        "in your dataset."
    )

    raise SystemExit


# ==========================================
# MODEL PREDICTIONS
# ==========================================

print(
    "\nRunning model predictions..."
)

predictions = model.predict(
    X
)


# ==========================================
# CONVERT PREDICTIONS
# ==========================================

predictions = pd.Series(
    predictions
)


# ==========================================
# HANDLE STRING MODEL PREDICTIONS
# ==========================================

predictions_numeric = predictions.apply(
    convert_label
)


# ==========================================
# REMOVE INVALID PREDICTIONS
# ==========================================

valid_predictions = (
    ~predictions_numeric.isna()
)


y_numeric = y_numeric.iloc[
    valid_predictions.values
]


predictions_numeric = (
    predictions_numeric[
        valid_predictions
    ]
    .astype(int)
)


# ==========================================
# FINAL SAFETY CHECK
# ==========================================

if len(
    predictions_numeric
) == 0:

    print(
        "\nERROR: Model produced no valid predictions."
    )

    raise SystemExit


# ==========================================
# MODEL ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_numeric,
    predictions_numeric
)


# ==========================================
# PRECISION
# ==========================================

precision = precision_score(
    y_numeric,
    predictions_numeric,
    average="weighted",
    zero_division=0
)


# ==========================================
# RECALL
# ==========================================

recall = recall_score(
    y_numeric,
    predictions_numeric,
    average="weighted",
    zero_division=0
)


# ==========================================
# F1 SCORE
# ==========================================

f1 = f1_score(
    y_numeric,
    predictions_numeric,
    average="weighted",
    zero_division=0
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

matrix = confusion_matrix(
    y_numeric,
    predictions_numeric,
    labels=[
        0,
        1,
        2,
        3
    ]
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_numeric,
    predictions_numeric,
    labels=[
        0,
        1,
        2,
        3
    ],
    target_names=LABEL_NAMES,
    zero_division=0
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print(
    "\n========================================"
)

print(
    "TRAFFIC AI MODEL PERFORMANCE"
)

print(
    "========================================"
)


print(
    f"\nAccuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print(
    "\n========================================"
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "========================================"
)

print(
    report
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

print(
    "\n========================================"
)

print(
    "CONFUSION MATRIX"
)

print(
    "========================================"
)

print(
    matrix
)


# ==========================================
# SAVE METRICS
# ==========================================

metrics = {

    "accuracy": float(
        accuracy
    ),

    "precision": float(
        precision
    ),

    "recall": float(
        recall
    ),

    "f1_score": float(
        f1
    ),

    "confusion_matrix": matrix.tolist(),

    "labels": LABEL_NAMES,

    "evaluation_samples": int(
        len(y_numeric)
    )

}


joblib.dump(
    metrics,
    METRICS_PATH
)


# ==========================================
# CONFIRM SAVE
# ==========================================

print(
    "\n✓ Model metrics saved to:"
)

print(
    METRICS_PATH
)


print(
    "\n========================================"
)

print(
    "MODEL EVALUATION COMPLETE"
)

print(
    "========================================"
)