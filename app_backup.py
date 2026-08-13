import streamlit as st
import pandas as pd
import time

from sklearn.ensemble import RandomForestClassifier


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Traffic Intelligence",
    page_icon="🚦",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("🚦 AI Traffic Intelligence System")

st.write(
    "AI-powered traffic monitoring, congestion prediction "
    "and intelligent signal optimization."
)


# ==========================================
# LOAD DATA
# ==========================================

data = pd.read_csv(
    "data/traffic_ml_dataset.csv"
)


# ==========================================
# TRAIN MODEL
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


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


# ==========================================
# CONGESTION LABELS
# ==========================================

congestion_names = {

    0: "Low",

    1: "Moderate",

    2: "High",

    3: "Critical"
}


# ==========================================
# GET LATEST ROUTE DATA
# ==========================================

latest_routes = (
    data
    .sort_values("Time_Seconds")
    .groupby("Route")
    .tail(1)
    .copy()
)


# ==========================================
# PREDICT
# ==========================================

latest_routes["Predicted_Code"] = model.predict(
    latest_routes[features]
)


latest_routes["Predicted_Congestion"] = (
    latest_routes["Predicted_Code"]
    .map(congestion_names)
)


# ==========================================
# SIGNAL OPTIMIZATION
# ==========================================

total_vehicles = latest_routes[
    "Total_Vehicles"
].sum()


latest_routes["Traffic_Share"] = (
    latest_routes["Total_Vehicles"]
    / total_vehicles
)


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
# DASHBOARD SUMMARY
# ==========================================

st.subheader("📊 Traffic Overview")


total = int(
    latest_routes["Total_Vehicles"].sum()
)

routes = len(latest_routes)

high_routes = len(
    latest_routes[
        latest_routes["Predicted_Code"] >= 2
    ]
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "🚗 Total Vehicles",
        total
    )


with col2:

    st.metric(
        "🛣️ Active Routes",
        routes
    )


with col3:

    st.metric(
        "⚠️ High/Critical Routes",
        high_routes
    )


# ==========================================
# ROUTE CARDS
# ==========================================

st.subheader("🛣️ Route Intelligence")


columns = st.columns(
    len(latest_routes)
)


for column, (_, row) in zip(
    columns,
    latest_routes.iterrows()
):

    with column:

        st.markdown(
            f"### {row['Route']}"
        )

        st.metric(
            "Vehicles",
            int(row["Total_Vehicles"])
        )

        st.metric(
            "Density",
            f"{row['Density'] * 100:.1f}%"
        )

        congestion = (
            row["Predicted_Congestion"]
        )

        if congestion == "Low":

            st.success(
                f"🟢 {congestion}"
            )

        elif congestion == "Moderate":

            st.warning(
                f"🟠 {congestion}"
            )

        else:

            st.error(
                f"🔴 {congestion}"
            )

        st.metric(
            "Recommended Green Time",
            f"{row['Green_Time']} sec"
        )


# ==========================================
# ROUTE COMPARISON
# ==========================================

st.subheader("📈 Route Comparison")


chart_data = latest_routes[
    ["Route", "Total_Vehicles"]
].set_index("Route")


st.bar_chart(
    chart_data
)


# ==========================================
# TRAFFIC DENSITY
# ==========================================

st.subheader("📊 Traffic Density")


density_data = latest_routes[
    ["Route", "Density"]
].copy()


density_data["Density"] *= 100

density_data = density_data.set_index(
    "Route"
)


st.bar_chart(
    density_data
)


# ==========================================
# RAW DATA
# ==========================================

with st.expander(
    "🔍 View Traffic Dataset"
):

    st.dataframe(
        data,
        use_container_width=True
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Traffic Intelligence System | "
    "YOLO + Machine Learning + Streamlit"
)