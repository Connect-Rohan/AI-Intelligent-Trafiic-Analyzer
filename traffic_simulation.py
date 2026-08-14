import streamlit as st


def traffic_simulation_controls(
    default_route_a=23,
    default_route_b=13
):

    st.subheader("🎛️ Interactive Traffic Simulation")

    st.write(
        "Adjust traffic and observe how the AI traffic system responds."
    )

    # ==========================================
    # ROUTE A
    # ==========================================

    st.markdown("### 🛣️ Route A")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cars_a = st.number_input(
            "🚗 Cars",
            min_value=0,
            value=default_route_a,
            step=1,
            key="cars_a"
        )

    with col2:
        motorcycles_a = st.number_input(
            "🏍️ Motorcycles",
            min_value=0,
            value=0,
            step=1,
            key="motorcycles_a"
        )

    with col3:
        buses_a = st.number_input(
            "🚌 Buses",
            min_value=0,
            value=0,
            step=1,
            key="buses_a"
        )

    with col4:
        trucks_a = st.number_input(
            "🚛 Trucks",
            min_value=0,
            value=0,
            step=1,
            key="trucks_a"
        )

    total_a = (
        cars_a
        + motorcycles_a
        + buses_a
        + trucks_a
    )

    st.metric(
        "Route A Total Vehicles",
        total_a
    )

    # ==========================================
    # ROUTE B
    # ==========================================

    st.markdown("### 🛣️ Route B")

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        cars_b = st.number_input(
            "🚗 Cars",
            min_value=0,
            value=default_route_b,
            step=1,
            key="cars_b"
        )

    with col6:
        motorcycles_b = st.number_input(
            "🏍️ Motorcycles",
            min_value=0,
            value=0,
            step=1,
            key="motorcycles_b"
        )

    with col7:
        buses_b = st.number_input(
            "🚌 Buses",
            min_value=0,
            value=0,
            step=1,
            key="buses_b"
        )

    with col8:
        trucks_b = st.number_input(
            "🚛 Trucks",
            min_value=0,
            value=0,
            step=1,
            key="trucks_b"
        )

    total_b = (
        cars_b
        + motorcycles_b
        + buses_b
        + trucks_b
    )

    st.metric(
        "Route B Total Vehicles",
        total_b
    )

    # ==========================================
    # EMERGENCY VEHICLE
    # ==========================================

    st.divider()

    st.markdown("### 🚨 Emergency Vehicle")

    emergency_route = st.selectbox(
        "Emergency vehicle approaching:",
        [
            "None",
            "Route A",
            "Route B"
        ]
    )

    emergency_active = st.checkbox(
        "🚨 Activate Emergency Priority"
    )

    # ==========================================
    # RETURN DATA
    # ==========================================

    return {
        "Route A": {
            "Cars": cars_a,
            "Motorcycles": motorcycles_a,
            "Buses": buses_a,
            "Trucks": trucks_a,
            "Total_Vehicles": total_a
        },

        "Route B": {
            "Cars": cars_b,
            "Motorcycles": motorcycles_b,
            "Buses": buses_b,
            "Trucks": trucks_b,
            "Total_Vehicles": total_b
        },

        "Emergency": emergency_active,

        "Emergency_Route": emergency_route
    }

# ==========================================
# TRAFFIC DENSITY
# ==========================================

def calculate_density(total_vehicles):

    # Current road capacity used by the project
    road_capacity = 50

    density = total_vehicles / road_capacity

    # Keep density between 0 and 1
    density = max(0, min(density, 1))

    return density


# ==========================================
# ML PREDICTION
# ==========================================

def predict_congestion(
    model,
    cars,
    motorcycles,
    buses,
    trucks
):

    total_vehicles = (
        cars
        + motorcycles
        + buses
        + trucks
    )

    density = calculate_density(
        total_vehicles
    )

    features = [[
        cars,
        motorcycles,
        buses,
        trucks,
        total_vehicles,
        density
    ]]

    prediction = model.predict(
        features
    )[0]

    return prediction, total_vehicles, density

# ==========================================
# CONGESTION LABEL
# ==========================================

def congestion_label(prediction):

    congestion_map = {

        0: "Low",

        1: "Moderate",

        2: "High",

        3: "Critical"

    }

    return congestion_map.get(
        int(prediction),
        "Unknown"
    )

# ==========================================
# TRAFFIC DENSITY
# ==========================================

def calculate_density(total_vehicles):

    # Road capacity used by our current model
    road_capacity = 50

    density = total_vehicles / road_capacity

    # Keep density between 0 and 1
    density = max(0, min(density, 1))

    return density


# ==========================================
# ML PREDICTION
# ==========================================

def predict_congestion(
    model,
    cars,
    motorcycles,
    buses,
    trucks
):

    total_vehicles = (
        cars
        + motorcycles
        + buses
        + trucks
    )

    density = calculate_density(
        total_vehicles
    )

    features = [[
        cars,
        motorcycles,
        buses,
        trucks,
        total_vehicles,
        density
    ]]

    prediction = model.predict(
        features
    )[0]

    return prediction, total_vehicles, density


# ==========================================
# CONGESTION LABEL
# ==========================================

def congestion_label(prediction):

    congestion_map = {
        0: "Low",
        1: "Moderate",
        2: "High",
        3: "Critical"
    }

    return congestion_map.get(
        int(prediction),
        "Unknown"
    )       