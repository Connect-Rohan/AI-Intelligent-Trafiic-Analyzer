import streamlit as st
import pandas as pd
from signal_optimizer import optimize_signals
import os
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from traffic_simulation import (
    traffic_simulation_controls,
    predict_congestion,
    congestion_label
)
import joblib

# ==========================================
# LOAD TRAINED TRAFFIC MODEL
# ==========================================

model = joblib.load(
    "traffic_model.pkl"
)

# ==========================================
# PAGE CONFIG
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

try:

    data = pd.read_csv(
        "data/traffic_ml_dataset.csv"
    )

except Exception as e:

    st.error(
        f"Could not load traffic dataset: {e}"
    )

    st.stop()


# ==========================================
# LATEST ROUTE DATA
# ==========================================

latest_routes = (
    data
    .sort_values("Time_Seconds")
    .groupby("Route")
    .tail(1)
    .copy()
)


# ==========================================
# TRAFFIC SHARE
# ==========================================

total_vehicles = latest_routes[
    "Total_Vehicles"
].sum()


if total_vehicles > 0:

    latest_routes["Traffic_Share"] = (
        latest_routes["Total_Vehicles"]
        / total_vehicles
    )

else:

    latest_routes["Traffic_Share"] = 0


# ==========================================
# SIGNAL OPTIMIZATION
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
# OVERVIEW
# ==========================================

st.subheader("📊 Traffic Overview")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🚗 Total Vehicles",
        int(total_vehicles)
    )


with col2:

    st.metric(
        "🛣️ Active Routes",
        len(latest_routes)
    )


with col3:

    average_density = (
        latest_routes["Density"].mean()
        * 100
    )

    st.metric(
        "📊 Average Density",
        f"{average_density:.1f}%"
    )


with col4:

    highest_route = latest_routes.loc[
        latest_routes["Total_Vehicles"].idxmax()
    ]

    st.metric(
        "⚠️ Highest Traffic",
        highest_route["Route"]
    )

# ==========================================
# DASHBOARD MODE
# ==========================================

st.divider()

mode = st.radio(
    "🖥 Dashboard Mode",
    [
        "📹 Camera Monitoring",
        "🎮 Simulation Dashboard"
    ],
    horizontal=True
)

st.divider()
if mode == "📹 Camera Monitoring":

    # ==========================================
    # AI TRAFFIC CAMERA FEEDS
    # ==========================================

    st.divider()

    st.subheader("🎥 AI Traffic Camera Feeds")

    video_col1, video_col2 = st.columns(2)

    # ==========================================
    # ROUTE A
    # ==========================================

    with video_col1:

        st.markdown("### 🛣️ Route A")

        route_a_video = "processed_videos/route_a_processed.mp4"

        if os.path.exists(route_a_video):

            st.video(
                route_a_video,
                format="video/mp4"
            )

        else:

            st.warning(
                "Route A processed video not found."
            )

    # ==========================================
    # ROUTE B
    # ==========================================

    with video_col2:

        st.markdown("### 🛣️ Route B")

        route_b_video = "processed_videos/route_b_processed.mp4"

        if os.path.exists(route_b_video):

            st.video(
                route_b_video,
                format="video/mp4"
            )

        else:

            st.warning(
                "Route B processed video not found."
            )

    # ==========================================
    # ROUTE INTELLIGENCE
    # ==========================================

    st.divider()

    st.subheader("🧠 AI Route Intelligence")

    route_columns = st.columns(len(latest_routes))

    for column, (_, route) in zip(
        route_columns,
        latest_routes.iterrows()
    ):

        with column:

            st.markdown(f"## {route['Route']}")

            st.metric(
                "🚗 Vehicles",
                int(route["Total_Vehicles"])
            )

            st.metric(
                "📊 Density",
                f"{route['Density'] * 100:.1f}%"
            )

            congestion = route["Congestion_Level"]

            if congestion == "Low":

                st.success(f"🟢 {congestion}")

            elif congestion == "Moderate":

                st.warning(f"🟠 {congestion}")

            elif congestion == "High":

                st.error(f"🔴 {congestion}")

            else:

                st.error(f"🚨 {congestion}")

            st.metric(
                "🚦 Green Time",
                f"{route['Green_Time']} sec"
            )

            st.metric(
                "📈 Traffic Share",
                f"{route['Traffic_Share'] * 100:.1f}%"
            )

    # ==========================================
    # TRAFFIC COMPARISON
    # ==========================================

    st.divider()

    st.subheader("📈 Traffic Comparison")

    comparison = latest_routes[
        ["Route", "Total_Vehicles"]
    ].set_index("Route")

    st.bar_chart(comparison)

    # ==========================================
    # TRAFFIC DENSITY
    # ==========================================

    st.subheader("📊 Traffic Density")

    density = latest_routes[
        ["Route", "Density"]
    ].copy()

    density["Density"] *= 100

    density = density.set_index("Route")

    st.line_chart(density)

    # ==========================================
    # TRAFFIC DATASET
    # ==========================================

    st.divider()

    with st.expander("🔍 View Complete Traffic Dataset"):

        st.dataframe(
            data,
            use_container_width=True
        )

elif mode == "🎮 Simulation Dashboard":

    st.subheader("🎮 AI Traffic Simulation Dashboard")

    top_left, top_right = st.columns([1, 2])
    bottom_left, bottom_right = st.columns([1, 2])

    # ==========================================
    # INTERACTIVE TRAFFIC SIMULATION
    # ==========================================    

    st.divider()

with top_left:

    st.subheader("🎮 Simulation Controls")

    simulation = traffic_simulation_controls(
        default_route_a=23,
        default_route_b=13
    )
    # ==========================================
    # REAL-TIME ROUTE A PREDICTION
    # ==========================================

    route_a = simulation["Route A"]

    prediction_a, total_a, density_a = predict_congestion(
        model,
        route_a["Cars"],
        route_a["Motorcycles"],
        route_a["Buses"],
        route_a["Trucks"]
    )

    congestion_a = congestion_label(
        prediction_a
    )


    # ==========================================
    # REAL-TIME ROUTE B PREDICTION
    # ==========================================

    route_b = simulation["Route B"]

    prediction_b, total_b, density_b = predict_congestion(
        model,
        route_b["Cars"],
        route_b["Motorcycles"],
        route_b["Buses"],
        route_b["Trucks"]
    )

    congestion_b = congestion_label(
        prediction_b
    )

    # ==========================================
    # AI SIGNAL OPTIMIZATION
    # ==========================================

    route_a_live = {
        "Total_Vehicles": total_a
    }

    route_b_live = {
        "Total_Vehicles": total_b
    }

    signal_data = optimize_signals(
        route_a_live,
        route_b_live
    )

    green_a = signal_data["Route A"]["Green_Time"]
    green_b = signal_data["Route B"]["Green_Time"]

    share_a = signal_data["Route A"]["Traffic_Share"]
    share_b = signal_data["Route B"]["Traffic_Share"]


    # ==========================================
    # CUSTOM TRAFFIC MAP
    # ==========================================

    st.divider()

    st.subheader("🗺️ AI Traffic Network")



    # ==========================================
    # LIVE ROUTE DATA
    # ==========================================

    route_data = {

        "Route A": {

            "vehicles": total_a,

            "density": density_a,

            "congestion": congestion_a,

            "green_time": green_a

        },

        "Route B": {

            "vehicles": total_b,

            "density": density_b,

            "congestion": congestion_b,

            "green_time": green_b

        }

    }

    # ==========================================
    # CONGESTION COLORS
    # ==========================================

    def get_route_color(level):

        if level == "Low":
            return "green"

        elif level == "Moderate":
            return "orange"

        elif level == "High":
            return "red"

        else:
            return "darkred"


    # ==========================================
    # CREATE MAP
    # ==========================================

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )


    # ==========================================
    # ROAD POSITIONS
    # ==========================================

    # Route A - horizontal
    route_a_x = [0, 5]
    route_a_y = [3, 3]

    # Route B - horizontal
    route_b_x = [5, 10]
    route_b_y = [3, 3]

    # Main vertical road
    main_x = [5, 5]
    main_y = [0, 6]


    # ==========================================
    # DRAW MAIN ROADS
    # ==========================================

    ax.plot(
        main_x,
        main_y,
        linewidth=18,
        color="lightgray",
        solid_capstyle="round"
    )


    # Route A
    if "Route A" in route_data:

        color_a = get_route_color(
            route_data["Route A"]["congestion"]
        )

        ax.plot(
            route_a_x,
            route_a_y,
            linewidth=18,
            color=color_a,
            solid_capstyle="round"
        )


    # Route B
    if "Route B" in route_data:

        color_b = get_route_color(
            route_data["Route B"]["congestion"]
        )

        ax.plot(
            route_b_x,
            route_b_y,
            linewidth=18,
            color=color_b,
            solid_capstyle="round"
        )


    # ==========================================
    # JUNCTION
    # ==========================================

    ax.scatter(
        5,
        3,
        s=1000,
        color="black",
        zorder=5
    )


    ax.text(
        5,
        3,
        "🚦",
        fontsize=24,
        ha="center",
        va="center",
        zorder=6
    )


    # ==========================================
    # ROUTE LABELS
    # ==========================================

    ax.text(
        2.5,
        3.45,
        "ROUTE A",
        fontsize=14,
        fontweight="bold",
        ha="center"
    )


    ax.text(
        7.5,
        3.45,
        "ROUTE B",
        fontsize=14,
        fontweight="bold",
        ha="center"
    )


    # ==========================================
    # ROUTE INFORMATION
    # ==========================================

    if "Route A" in route_data:

        route = route_data["Route A"]

        ax.text(
            2.5,
            2.45,
            f"🚗 {route['vehicles']} vehicles\n"
            f"Density: {route['density'] * 100:.1f}%\n"
            f"{route['congestion']}\n"
            f"Green: {route['green_time']} sec",
            ha="center",
            fontsize=10
        )


    if "Route B" in route_data:

        route = route_data["Route B"]

        ax.text(
            7.5,
            2.45,
            f"🚗 {route['vehicles']} vehicles\n"
            f"Density: {route['density'] * 100:.1f}%\n"
            f"{route['congestion']}\n"
            f"Green: {route['green_time']} sec",
            ha="center",
            fontsize=10
        )


    # ==========================================
    # NORTH / SOUTH ROADS
    # ==========================================

    ax.text(
        5,
        5.7,
        "⬆️ NORTH",
        ha="center",
        fontsize=11
    )


    ax.text(
        5,
        0.3,
        "⬇️ SOUTH",
        ha="center",
        fontsize=11
    )


    # ==========================================
    # MAP SETTINGS
    # ==========================================

    ax.set_xlim(
        -1,
        11
    )

    ax.set_ylim(
        -1,
        7
    )

    ax.axis(
        "off"
    )

    ax.set_title(
        "AI-Powered Traffic Network",
        fontsize=18,
        fontweight="bold"
    )


    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    # ==========================================
    # LIVE TRAFFIC SIGNAL SIMULATION
    # ==========================================

    st.divider()

    st.subheader("🚦 Live AI Traffic Signal Control")

    st.write(
        "The traffic signals continuously operate using "
        "green-time recommendations generated by the AI system."
    )


    # ==========================================
    # GET GREEN TIMES
    # ==========================================

    route_a_green = int(
        route_data.get(
            "Route A",
            {}
        ).get(
            "green_time",
            30
        )
    )

    route_b_green = int(
        route_data.get(
            "Route B",
            {}
        ).get(
            "green_time",
            30
        )
    )


    # ==========================================
    # LIMIT EXTREME VALUES
    # ==========================================

    route_a_green = max(
        10,
        min(route_a_green, 90)
    )

    route_b_green = max(
        10,
        min(route_b_green, 90)
    )


    # ==========================================
    # YELLOW TIME
    # ==========================================

    yellow_time = 3


    # ==========================================
    # TOTAL CYCLE
    # ==========================================

    total_cycle = (
        route_a_green
        + yellow_time
        + route_b_green
        + yellow_time
    )


    # ==========================================
    # LIVE SIGNAL HTML
    # ==========================================

    signal_html = f"""

    <!DOCTYPE html>

    <html>

    <head>

    <style>

    body {{

        margin: 0;

        font-family:
            Arial,
            sans-serif;

        background:
            transparent;

    }}


    .container {{

        width: 100%;

        background:
            #111827;

        border-radius:
            18px;

        padding:
            25px;

        box-sizing:
            border-box;

        color:
            white;

    }}


    .title {{

        text-align:
            center;

        font-size:
            24px;

        font-weight:
            bold;

        margin-bottom:
            25px;

    }}


    .roads {{

        display:
            flex;

        align-items:
            center;

        justify-content:
            space-between;

        gap:
            15px;

    }}


    .route {{

        flex:
            1;

        background:
            #1f2937;

        border-radius:
            15px;

        padding:
            20px;

        text-align:
            center;

    }}


    .route-name {{

        font-size:
            20px;

        font-weight:
            bold;

        margin-bottom:
            15px;

    }}


    .road {{

        height:
            65px;

        border-radius:
            10px;

        display:
            flex;

        align-items:
            center;

        justify-content:
            center;

        font-size:
            18px;

        font-weight:
            bold;

        transition:
            background 0.5s;

    }}


    .signal-box {{

        width:
            85px;

        background:
            #111;

        border-radius:
            20px;

        padding:
            12px;

        display:
            flex;

        flex-direction:
            column;

        align-items:
            center;

        gap:
            10px;

    }}


    .light {{

        width:
            35px;

        height:
            35px;

        border-radius:
            50%;

        background:
            #333;

        transition:
            0.4s;

    }}


    .active-red {{

        background:
            #ef4444;

        box-shadow:
            0 0 25px
            #ef4444;

    }}


    .active-yellow {{

        background:
            #facc15;

        box-shadow:
            0 0 25px
            #facc15;

    }}


    .active-green {{

        background:
            #22c55e;

        box-shadow:
            0 0 25px
            #22c55e;

    }}


    .status {{

        margin-top:
            15px;

        font-size:
            18px;

        font-weight:
            bold;

    }}


    .timer {{

        font-size:
            34px;

        font-weight:
            bold;

        margin-top:
            8px;

    }}


    .info {{

        margin-top:
            25px;

        text-align:
            center;

        color:
            #d1d5db;

    }}


    </style>

    </head>


    <body>


    <div class="container">


    <div class="title">

    🚦 AI Traffic Signal Controller

    </div>


    <div class="roads">


    <!-- ROUTE A -->

    <div class="route">

    <div class="route-name">

    🛣️ Route A

    </div>


    <div
        id="roadA"
        class="road"
    >

    ROUTE A

    </div>


    <div
        id="statusA"
        class="status"
    >

    WAITING

    </div>


    <div
        id="timerA"
        class="timer"
    >

    --

    </div>

    </div>


    <!-- SIGNAL -->

    <div class="signal-box">

    <div
        id="red"
        class="light"
    >
    </div>

    <div
        id="yellow"
        class="light"
    >
    </div>

    <div
        id="green"
        class="light"
    >
    </div>

    </div>


    <!-- ROUTE B -->

    <div class="route">

    <div class="route-name">

    🛣️ Route B

    </div>


    <div
        id="roadB"
        class="road"
    >

    ROUTE B

    </div>


    <div
        id="statusB"
        class="status"
    >

    WAITING

    </div>


    <div
        id="timerB"
        class="timer"
    >

    --

    </div>

    </div>


    </div>


    <div class="info">

    AI Recommended Green Time:
    Route A = {route_a_green}s
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Route B = {route_b_green}s

    </div>


    </div>


    <script>


    // ========================================
    // SIGNAL TIMINGS
    // ========================================

    const routeAGreen =
        {route_a_green};

    const routeBGreen =
        {route_b_green};

    const yellowTime =
        {yellow_time};


    // ========================================
    // DOM ELEMENTS
    // ========================================

    const red =
        document.getElementById("red");

    const yellow =
        document.getElementById("yellow");

    const green =
        document.getElementById("green");


    const roadA =
        document.getElementById("roadA");

    const roadB =
        document.getElementById("roadB");


    const statusA =
        document.getElementById("statusA");

    const statusB =
        document.getElementById("statusB");


    const timerA =
        document.getElementById("timerA");

    const timerB =
        document.getElementById("timerB");


    // ========================================
    // RESET SIGNAL
    // ========================================

    function resetLights() {{

        red.className =
            "light";

        yellow.className =
            "light";

        green.className =
            "light";

    }}


    // ========================================
    // ROUTE A GREEN
    // ========================================

    function routeAGreenPhase(seconds) {{

        resetLights();

        green.className =
            "light active-green";


        roadA.style.background =
            "#166534";

        roadB.style.background =
            "#7f1d1d";


        statusA.innerHTML =
            "🟢 GREEN";

        statusB.innerHTML =
            "🔴 STOP";


        let remaining =
            seconds;


        timerA.innerHTML =
            remaining + " sec";


        timerB.innerHTML =
            "WAIT";


        const interval =
            setInterval(() => {{

            remaining--;

            if (
                remaining <= 0
            ) {{

                clearInterval(interval);

                routeAYellowPhase();

            }}

            else {{

                timerA.innerHTML =
                    remaining + " sec";

            }}

        }}, 1000);

    }}


    // ========================================
    // ROUTE A YELLOW
    // ========================================

    function routeAYellowPhase() {{

        resetLights();

        yellow.className =
            "light active-yellow";


        roadA.style.background =
            "#a16207";


        roadB.style.background =
            "#7f1d1d";


        statusA.innerHTML =
            "🟡 CHANGE";


        statusB.innerHTML =
            "🔴 STOP";


        let remaining =
            yellowTime;


        timerA.innerHTML =
            remaining + " sec";


        const interval =
            setInterval(() => {{

            remaining--;


            if (
                remaining <= 0
            ) {{

                clearInterval(interval);

                routeBGreenPhase();

            }}

            else {{

                timerA.innerHTML =
                    remaining + " sec";

            }}

        }}, 1000);

    }}


    // ========================================
    // ROUTE B GREEN
    // ========================================

    function routeBGreenPhase(seconds) {{

        resetLights();

        green.className =
            "light active-green";


        roadA.style.background =
            "#7f1d1d";

        roadB.style.background =
            "#166534";


        statusA.innerHTML =
            "🔴 STOP";

        statusB.innerHTML =
            "🟢 GREEN";


        let remaining =
            seconds;


        timerA.innerHTML =
            "WAIT";


        timerB.innerHTML =
            remaining + " sec";


        const interval =
            setInterval(() => {{

            remaining--;


            if (
                remaining <= 0
            ) {{

                clearInterval(interval);

                routeBYellowPhase();

            }}

            else {{

                timerB.innerHTML =
                    remaining + " sec";

            }}

        }}, 1000);

    }}


    // ========================================
    // ROUTE B YELLOW
    // ========================================

    function routeBYellowPhase() {{

        resetLights();

        yellow.className =
            "light active-yellow";


        roadA.style.background =
            "#7f1d1d";

        roadB.style.background =
            "#a16207";


        statusA.innerHTML =
            "🔴 STOP";

        statusB.innerHTML =
            "🟡 CHANGE";


        let remaining =
            yellowTime;


        timerB.innerHTML =
            remaining + " sec";


        const interval =
            setInterval(() => {{

            remaining--;


            if (
                remaining <= 0
            ) {{

                clearInterval(interval);

                routeAGreenPhase(
                    routeAGreen
                );

            }}

            else {{

                timerB.innerHTML =
                    remaining + " sec";

            }}

        }}, 1000);

    }}


    // ========================================
    // START SYSTEM
    // ========================================

    routeAGreenPhase(
        routeAGreen
    );


    </script>


    </body>

    </html>

    """


    # ==========================================
    # DISPLAY COMPONENT
    # ==========================================

    components.html(
        signal_html,
        height=430,
        scrolling=False
    )



    # ==========================================
    # REAL-TIME AI RESULTS
    # ==========================================

    st.divider()

    st.subheader("🤖 Real-Time AI Traffic Analysis")

    col1, col2 = st.columns(2)


    with col1:

        st.markdown("### 🛣️ Route A")

        st.metric(
            "Total Vehicles",
            total_a
        )

        st.metric(
            "Density",
            f"{density_a * 100:.1f}%"
        )

        st.metric(
            "AI Congestion",
            congestion_a
        )

        st.metric(
        "Traffic Share",
        f"{share_a*100:.1f}%"
        )

        st.metric(
        "AI Green Time",
        f"{green_a} sec"
        )


    with col2:

        st.markdown("### 🛣️ Route B")

        st.metric(
            "Total Vehicles",
            total_b
        )

        st.metric(
            "Density",
            f"{density_b * 100:.1f}%"
        )

        st.metric(
            "AI Congestion",
            congestion_b
        )

        st.metric(
        "Traffic Share",
        f"{share_b*100:.1f}%"
        )

        st.metric(
        "AI Green Time",
        f"{green_b} sec"
        )

    # ==========================================
    # FOOTER
    # ==========================================

    st.divider()

    st.caption(
    "AI Traffic Intelligence System | "
    "YOLO + Computer Vision + Machine Learning + Streamlit"
)