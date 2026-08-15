import streamlit as st
import pandas as pd
from signal_optimizer import optimize_signals
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
# GLOBAL DASHBOARD CSS
# ==========================================

st.markdown("""
<style>

/* ---- Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ---- Root variables ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Main page background ---- */
.stApp {
    background: #f1f5f9;
}

/* ---- Hide default Streamlit top decoration ---- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ---- Section header style ---- */
.dash-header {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid #cbd5e1;
}

/* ---- Glowing metric card ---- */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.metric-card:hover {
    border-color: #3b82f6;
}
.metric-card .mc-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 4px;
}
.metric-card .mc-value {
    font-size: 26px;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
}
.metric-card .mc-sub {
    font-size: 12px;
    color: #64748b;
    margin-top: 2px;
}

/* ---- Overview metric cards ---- */
.ov-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.ov-card .ov-icon {
    font-size: 28px;
    line-height: 1;
    margin-bottom: 8px;
}
.ov-card .ov-val {
    font-size: 30px;
    font-weight: 800;
    color: #0f172a;
}
.ov-card .ov-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #64748b;
    margin-top: 4px;
}

/* ---- AI Analysis route cards ---- */
.route-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 22px 24px;
    height: 100%;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.route-card .rc-title {
    font-size: 16px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid #e2e8f0;
}
.route-card .rc-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #f1f5f9;
}
.route-card .rc-key {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.8px;
    color: #64748b;
    text-transform: uppercase;
}
.route-card .rc-val {
    font-size: 15px;
    font-weight: 700;
    color: #1e293b;
}

/* ---- Congestion badge ---- */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-low    { background: #dcfce7; color: #16a34a; }
.badge-moderate { background: #ffedd5; color: #c2410c; }
.badge-high   { background: #fee2e2; color: #dc2626; }
.badge-critical { background: #fecaca; color: #991b1b; }

/* ---- Panel container ---- */
.panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* ---- Streamlit default metric overrides ---- */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 12px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase;
    color: #64748b !important;
}
[data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 800 !important;
    color: #0f172a !important;
}

/* ---- Dashboard title ---- */
.dash-title {
    font-size: 32px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}
.dash-subtitle {
    font-size: 14px;
    color: #64748b;
    margin-bottom: 24px;
}

/* ---- Simulation section title ---- */
.sim-section-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ---- Route sub-panel ---- */
.route-panel {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.route-panel-title {
    font-size: 13px;
    font-weight: 700;
    color: #475569;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* ---- Divider ---- */
hr {
    border-color: #e2e8f0 !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# TITLE
# ==========================================

st.markdown("""
<div class="dash-title">🚦 AI Traffic Intelligence System</div>
<div class="dash-subtitle">AI-powered traffic monitoring, congestion prediction and intelligent signal optimization.</div>
""", unsafe_allow_html=True)


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

st.markdown('<div class="dash-header">📊 Traffic Overview</div>', unsafe_allow_html=True)

average_density = (
    latest_routes["Density"].mean()
    * 100
)

highest_route = latest_routes.loc[
    latest_routes["Total_Vehicles"].idxmax()
]

ov1, ov2, ov3, ov4 = st.columns(4)

with ov1:
    st.markdown(f"""
    <div class="ov-card">
        <div class="ov-icon">🚗</div>
        <div class="ov-val">{int(total_vehicles)}</div>
        <div class="ov-label">Total Vehicles</div>
    </div>
    """, unsafe_allow_html=True)

with ov2:
    st.markdown(f"""
    <div class="ov-card">
        <div class="ov-icon">🛣️</div>
        <div class="ov-val">{len(latest_routes)}</div>
        <div class="ov-label">Active Routes</div>
    </div>
    """, unsafe_allow_html=True)

with ov3:
    st.markdown(f"""
    <div class="ov-card">
        <div class="ov-icon">📊</div>
        <div class="ov-val">{average_density:.1f}%</div>
        <div class="ov-label">Avg Density</div>
    </div>
    """, unsafe_allow_html=True)

with ov4:
    st.markdown(f"""
    <div class="ov-card">
        <div class="ov-icon">⚠️</div>
        <div class="ov-val">{highest_route['Route']}</div>
        <div class="ov-label">Highest Traffic</div>
    </div>
    """, unsafe_allow_html=True)


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

    # ==========================================
    # SIMULATION DASHBOARD HEADER
    # ==========================================

    st.markdown("""
    <div style="margin-bottom:18px;">
        <div class="dash-header">🎮 AI Traffic Simulation Dashboard</div>
        <div style="font-size:13px; color:#475569;">
            Adjust live traffic inputs and watch the AI system adapt signals in real time.
        </div>
    </div>
    """, unsafe_allow_html=True)


    # ==========================================
    # TOP ROW — Controls | Network Map
    # ==========================================

    top_left, top_right = st.columns([1, 1.6], gap="large")

    with top_left:

        # ==========================================
        # INTERACTIVE TRAFFIC SIMULATION CONTROLS
        # ==========================================

        st.markdown('<div class="sim-section-title">🎛️ Simulation Controls</div>',
                    unsafe_allow_html=True)

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
            return "#22c55e"

        elif level == "Moderate":
            return "#f97316"

        elif level == "High":
            return "#ef4444"

        else:
            return "#7f1d1d"


    # ==========================================
    # TOP RIGHT — AI TRAFFIC NETWORK MAP
    # ==========================================

    with top_right:

        st.markdown('<div class="sim-section-title">🗺️ Live AI Traffic Network</div>',
                    unsafe_allow_html=True)

        # ==========================================
        # CREATE MAP — DARK PROFESSIONAL STYLE
        # ==========================================

        fig, ax = plt.subplots(figsize=(11, 5.5))

        # Dark background
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")


        # ==========================================
        # ROAD POSITIONS
        # ==========================================

        # Route A — horizontal left
        route_a_x = [0.3, 4.7]
        route_a_y = [3, 3]

        # Route B — horizontal right
        route_b_x = [5.3, 9.7]
        route_b_y = [3, 3]

        # Main vertical road through junction
        main_x = [5, 5]
        main_y = [0.3, 5.7]


        # ==========================================
        # DRAW ROAD SHADOWS (dark gray base)
        # ==========================================

        ax.plot(
            [0.2, 9.8], [3, 3],
            linewidth=30,
            color="#cbd5e1",
            solid_capstyle="round",
            zorder=1
        )

        ax.plot(
            main_x, main_y,
            linewidth=24,
            color="#cbd5e1",
            solid_capstyle="round",
            zorder=1
        )


        # ==========================================
        # DRAW ROADS — COLORED BY CONGESTION
        # ==========================================

        color_a = get_route_color(
            route_data["Route A"]["congestion"]
        )

        color_b = get_route_color(
            route_data["Route B"]["congestion"]
        )

        # Route A road
        ax.plot(
            route_a_x, route_a_y,
            linewidth=22,
            color=color_a,
            solid_capstyle="butt",
            alpha=0.85,
            zorder=2
        )

        # Route B road
        ax.plot(
            route_b_x, route_b_y,
            linewidth=22,
            color=color_b,
            solid_capstyle="butt",
            alpha=0.85,
            zorder=2
        )

        # Vertical cross road (neutral gray)
        ax.plot(
            main_x, main_y,
            linewidth=18,
            color="#94a3b8",
            solid_capstyle="butt",
            alpha=0.9,
            zorder=2
        )


        # ==========================================
        # CENTER LANE DIVIDERS (dashed white)
        # ==========================================

        ax.plot(
            [0.5, 4.5], [3, 3],
            linewidth=1.5,
            color="#64748b",
            linestyle="--",
            dashes=(6, 4),
            alpha=0.5,
            zorder=3
        )

        ax.plot(
            [5.5, 9.5], [3, 3],
            linewidth=1.5,
            color="#64748b",
            linestyle="--",
            dashes=(6, 4),
            alpha=0.5,
            zorder=3
        )

        ax.plot(
            [5, 5], [0.5, 2.6],
            linewidth=1.5,
            color="#64748b",
            linestyle="--",
            dashes=(5, 4),
            alpha=0.5,
            zorder=3
        )

        ax.plot(
            [5, 5], [3.4, 5.5],
            linewidth=1.5,
            color="#64748b",
            linestyle="--",
            dashes=(5, 4),
            alpha=0.5,
            zorder=3
        )


        # ==========================================
        # JUNCTION — CIRCLE + SIGNAL ICON
        # ==========================================

        junction_circle = plt.Circle(
            (5, 3),
            0.52,
            color="#e2e8f0",
            zorder=4
        )
        ax.add_patch(junction_circle)

        junction_ring = plt.Circle(
            (5, 3),
            0.52,
            fill=False,
            edgecolor="#fbbf24",
            linewidth=2.5,
            zorder=5
        )
        ax.add_patch(junction_ring)

        ax.text(
            5, 3, "🚦",
            fontsize=20,
            ha="center",
            va="center",
            zorder=6
        )


        # ==========================================
        # ROUTE LABELS (above roads)
        # ==========================================

        ax.text(
            2.5, 3.65,
            "ROUTE A",
            fontsize=11,
            fontweight="bold",
            color="#1e293b",
            ha="center",
            va="center",
            zorder=7
        )

        ax.text(
            7.5, 3.65,
            "ROUTE B",
            fontsize=11,
            fontweight="bold",
            color="#1e293b",
            ha="center",
            va="center",
            zorder=7
        )


        # ==========================================
        # ROUTE INFO BOXES (below roads)
        # ==========================================

        if "Route A" in route_data:

            route = route_data["Route A"]

            ax.text(
                2.5, 2.05,
                f"🚗 {route['vehicles']} vehicles   "
                f"Density: {route['density'] * 100:.1f}%\n"
                f"{route['congestion']}   "
                f"Green: {route['green_time']}s",
                ha="center",
                va="center",
                fontsize=8.5,
                color="#334155",
                zorder=7,
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="#ffffff",
                    edgecolor=color_a,
                    linewidth=1.5,
                    alpha=0.9
                )
            )

        if "Route B" in route_data:

            route = route_data["Route B"]

            ax.text(
                7.5, 2.05,
                f"🚗 {route['vehicles']} vehicles   "
                f"Density: {route['density'] * 100:.1f}%\n"
                f"{route['congestion']}   "
                f"Green: {route['green_time']}s",
                ha="center",
                va="center",
                fontsize=8.5,
                color="#cbd5e1",
                zorder=7,
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="#0f172a",
                    edgecolor=color_b,
                    linewidth=1.5,
                    alpha=0.9
                )
            )


        # ==========================================
        # NORTH / SOUTH LABELS
        # ==========================================

        ax.text(
            5, 5.55,
            "▲  NORTH",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#94a3b8",
            zorder=7
        )

        ax.text(
            5, 0.45,
            "▼  SOUTH",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#94a3b8",
            zorder=7
        )


        # ==========================================
        # LEGEND — CONGESTION COLORS
        # ==========================================

        legend_patches = [
            mpatches.Patch(color="#22c55e", label="Low"),
            mpatches.Patch(color="#f97316", label="Moderate"),
            mpatches.Patch(color="#ef4444", label="High"),
            mpatches.Patch(color="#7f1d1d", label="Critical"),
        ]

        legend = ax.legend(
            handles=legend_patches,
            loc="upper left",
            frameon=True,
            framealpha=0.95,
            facecolor="#ffffff",
            edgecolor="#e2e8f0",
            fontsize=8,
            title="Congestion",
            title_fontsize=8,
            labelcolor="#374151"
        )
        legend.get_title().set_color("#64748b")


        # ==========================================
        # MAP SETTINGS
        # ==========================================

        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 6.5)
        ax.axis("off")

        ax.set_title(
            "AI-Powered Traffic Network — Live View",
            fontsize=13,
            fontweight="bold",
            color="#475569",
            pad=12,
            loc="left"
        )

        fig.tight_layout(pad=1.0)

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # ==========================================
    # BOTTOM ROW — Signal Controller | AI Analysis
    # ==========================================

    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

    bottom_left, bottom_right = st.columns([1, 1.6], gap="large")


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
    # BOTTOM LEFT — LIVE SIGNAL HTML
    # ==========================================

    with bottom_left:

        st.markdown(
            '<div class="sim-section-title">🚦 AI Traffic Signal Controller</div>',
            unsafe_allow_html=True
        )

        signal_html = f"""

        <!DOCTYPE html>

        <html>

        <head>

        <style>

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', Arial, sans-serif;
            background: transparent;
        }}

        .container {{
            width: 100%;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 22px 20px 18px;
            color: #0f172a;
        }}

        .ctrl-title {{
            text-align: center;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 18px;
            padding-bottom: 14px;
            border-bottom: 1px solid #e2e8f0;
        }}

        .roads {{
            display: flex;
            align-items: stretch;
            justify-content: space-between;
            gap: 14px;
        }}

        .route {{
            flex: 1;
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 16px 14px;
            text-align: center;
            transition: border-color 0.4s;
        }}

        .route-name {{
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: #475569;
            margin-bottom: 12px;
        }}

        .road {{
            height: 56px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            transition: background 0.5s ease;
            background: #cbd5e1;
            color: #475569;
        }}

        /* ---- Signal housing ---- */
        .signal-box {{
            width: 72px;
            background: #e2e8f0;
            border: 2px solid #cbd5e1;
            border-radius: 36px;
            padding: 14px 10px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
            flex-shrink: 0;
            box-shadow: 0 4px 24px rgba(0,0,0,0.12);
        }}

        .light {{
            width: 42px;
            height: 42px;
            border-radius: 50%;
            background: #cbd5e1;
            transition: background 0.4s, box-shadow 0.4s;
        }}

        .active-red {{
            background: #ef4444;
            box-shadow: 0 0 20px #ef4444, 0 0 40px rgba(239,68,68,0.4);
        }}

        .active-yellow {{
            background: #facc15;
            box-shadow: 0 0 20px #facc15, 0 0 40px rgba(250,204,21,0.4);
        }}

        .active-green {{
            background: #22c55e;
            box-shadow: 0 0 20px #22c55e, 0 0 40px rgba(34,197,94,0.4);
        }}

        /* ---- Status badge ---- */
        .status {{
            margin-top: 12px;
            display: inline-block;
            padding: 4px 14px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            background: #e2e8f0;
            color: #64748b;
            transition: background 0.5s, color 0.5s;
        }}

        .status-green  {{ background: #dcfce7; color: #16a34a; }}
        .status-yellow {{ background: #fef9c3; color: #ca8a04; }}
        .status-red    {{ background: #fee2e2; color: #dc2626; }}

        /* ---- Timer ---- */
        .timer {{
            font-size: 36px;
            font-weight: 800;
            margin-top: 10px;
            font-family: 'Inter', monospace;
            color: #0f172a;
            letter-spacing: -1px;
            min-height: 44px;
            line-height: 1;
            transition: color 0.5s;
        }}

        .timer-green  {{ color: #16a34a; }}
        .timer-yellow {{ color: #ca8a04; }}
        .timer-red    {{ color: #dc2626; }}

        /* ---- Info footer ---- */
        .info {{
            margin-top: 16px;
            padding-top: 14px;
            border-top: 1px solid #e2e8f0;
            text-align: center;
            font-size: 11px;
            color: #64748b;
            letter-spacing: 0.5px;
        }}

        .info span {{
            color: #374151;
            font-weight: 600;
        }}

        </style>

        </head>


        <body>


        <div class="container">


        <div class="ctrl-title">
            🚦 &nbsp; Smart Signal Controller &nbsp; 🚦
        </div>


        <div class="roads">


        <!-- ROUTE A -->

        <div class="route" id="routeBoxA">

        <div class="route-name">Route A</div>

        <div
            id="roadA"
            class="road"
        >
        ← ROUTE A →
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

        <div class="route" id="routeBoxB">

        <div class="route-name">Route B</div>

        <div
            id="roadB"
            class="road"
        >
        ← ROUTE B →
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
            AI Recommended Green Time &nbsp;|&nbsp;
            Route A = <span>{route_a_green}s</span>
            &nbsp;&nbsp;·&nbsp;&nbsp;
            Route B = <span>{route_b_green}s</span>
            &nbsp;&nbsp;·&nbsp;&nbsp;
            Cycle = <span>{total_cycle}s</span>
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


        const routeBoxA =
            document.getElementById("routeBoxA");

        const routeBoxB =
            document.getElementById("routeBoxB");


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
                "#450a0a";


            statusA.innerHTML =
                "🟢 GREEN";

            statusA.className =
                "status status-green";

            statusB.innerHTML =
                "🔴 STOP";

            statusB.className =
                "status status-red";


            routeBoxA.style.borderColor =
                "#22c55e";

            routeBoxB.style.borderColor =
                "#ef4444";


            let remaining =
                seconds;


            timerA.innerHTML =
                remaining + "s";

            timerA.className =
                "timer timer-green";


            timerB.innerHTML =
                "WAIT";

            timerB.className =
                "timer timer-red";


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
                        remaining + "s";

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
                "#713f12";


            roadB.style.background =
                "#450a0a";


            statusA.innerHTML =
                "🟡 CHANGE";

            statusA.className =
                "status status-yellow";


            statusB.innerHTML =
                "🔴 STOP";

            statusB.className =
                "status status-red";


            routeBoxA.style.borderColor =
                "#facc15";


            let remaining =
                yellowTime;


            timerA.innerHTML =
                remaining + "s";

            timerA.className =
                "timer timer-yellow";


            const interval =
                setInterval(() => {{

                remaining--;


                if (
                    remaining <= 0
                ) {{

                    clearInterval(interval);

                    // Pass routeBGreen to fix the "undefined" bug
                    routeBGreenPhase(
                        routeBGreen
                    );

                }}

                else {{

                    timerA.innerHTML =
                        remaining + "s";

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
                "#450a0a";

            roadB.style.background =
                "#166534";


            statusA.innerHTML =
                "🔴 STOP";

            statusA.className =
                "status status-red";

            statusB.innerHTML =
                "🟢 GREEN";

            statusB.className =
                "status status-green";


            routeBoxA.style.borderColor =
                "#ef4444";

            routeBoxB.style.borderColor =
                "#22c55e";


            let remaining =
                seconds;


            timerA.innerHTML =
                "WAIT";

            timerA.className =
                "timer timer-red";


            timerB.innerHTML =
                remaining + "s";

            timerB.className =
                "timer timer-green";


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
                        remaining + "s";

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
                "#450a0a";

            roadB.style.background =
                "#713f12";


            statusA.innerHTML =
                "🔴 STOP";

            statusA.className =
                "status status-red";

            statusB.innerHTML =
                "🟡 CHANGE";

            statusB.className =
                "status status-yellow";


            routeBoxB.style.borderColor =
                "#facc15";


            let remaining =
                yellowTime;


            timerB.innerHTML =
                remaining + "s";

            timerB.className =
                "timer timer-yellow";


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
                        remaining + "s";

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
        # DISPLAY SIGNAL COMPONENT
        # ==========================================

        components.html(
            signal_html,
            height=440,
            scrolling=False
        )


    # ==========================================
    # BOTTOM RIGHT — REAL-TIME AI ANALYSIS CARDS
    # ==========================================

    with bottom_right:

        st.markdown(
            '<div class="sim-section-title">🤖 Real-Time AI Analysis</div>',
            unsafe_allow_html=True
        )

        # ==========================================
        # CONGESTION BADGE HELPER
        # ==========================================

        def congestion_badge(level):

            badge_class = {
                "Low":      "badge-low",
                "Moderate": "badge-moderate",
                "High":     "badge-high",
                "Critical": "badge-critical"
            }.get(level, "badge-high")

            return (
                f'<span class="badge {badge_class}">'
                f'{level}</span>'
            )

        # ==========================================
        # ROUTE A CARD
        # ==========================================

        badge_a = congestion_badge(congestion_a)

        left_border_a = get_route_color(congestion_a)

        st.markdown(f"""
        <div class="route-card" style="border-left: 4px solid {left_border_a}; margin-bottom: 16px;">
            <div class="rc-title">🛣️ Route A</div>
            <div class="rc-row">
                <span class="rc-key">🚗 Total Vehicles</span>
                <span class="rc-val">{total_a}</span>
            </div>
            <div class="rc-row">
                <span class="rc-key">📊 Density</span>
                <span class="rc-val">{density_a * 100:.1f}%</span>
            </div>
            <div class="rc-row">
                <span class="rc-key">⚡ Congestion</span>
                <span class="rc-val">{badge_a}</span>
            </div>
            <div class="rc-row">
                <span class="rc-key">📈 Traffic Share</span>
                <span class="rc-val">{share_a * 100:.1f}%</span>
            </div>
            <div class="rc-row" style="border-bottom: none;">
                <span class="rc-key">🚦 AI Green Time</span>
                <span class="rc-val">{green_a}s</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


        # ==========================================
        # ROUTE B CARD
        # ==========================================

        badge_b = congestion_badge(congestion_b)

        left_border_b = get_route_color(congestion_b)

        st.markdown(f"""
        <div class="route-card" style="border-left: 4px solid {left_border_b};">
            <div class="rc-title">🛣️ Route B</div>
            <div class="rc-row">
                <span class="rc-key">🚗 Total Vehicles</span>
                <span class="rc-val">{total_b}</span>
            </div>
            <div class="rc-row">
                <span class="rc-key">📊 Density</span>
                <span class="rc-val">{density_b * 100:.1f}%</span>
            </div>
            <div class="rc-row">
                <span class="rc-key">⚡ Congestion</span>
                <span class="rc-val">{badge_b}</span>
            </div>
            <div class="rc-row">
                <span class="rc-key">📈 Traffic Share</span>
                <span class="rc-val">{share_b * 100:.1f}%</span>
            </div>
            <div class="rc-row" style="border-bottom: none;">
                <span class="rc-key">🚦 AI Green Time</span>
                <span class="rc-val">{green_b}s</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


    # ==========================================
    # FOOTER
    # ==========================================

    st.divider()

    st.markdown("""
    <div style="text-align:center; font-size:11px; color:#334155; letter-spacing:1px; padding: 4px 0 8px;">
        AI Traffic Intelligence System &nbsp;|&nbsp;
        YOLO + Computer Vision + Machine Learning + Streamlit
    </div>
    """, unsafe_allow_html=True)