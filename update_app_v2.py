import os

with open("app.py", "r", encoding="utf-8") as f:
    original_code = f.read()

# 1. Insert AI Model Performance section right before Dashboard Mode
model_perf_block = """
# ==========================================
# AI MODEL PERFORMANCE
# ==========================================

try:
    if os.path.exists('model_metrics.pkl'):
        metrics = joblib.load('model_metrics.pkl')
        with st.expander("🔍 View AI Model Performance Details"):
            st.markdown("### ML Congestion-Classification Metrics")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
            m_col2.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
            m_col3.metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")
            m_col4.metric("F1 Score", f"{metrics.get('f1_score', 0)*100:.2f}%")
            
            st.markdown(f"**Evaluation Samples:** {metrics.get('evaluation_samples', 0)}")
            st.markdown(f"**Classes Evaluated:** {', '.join(metrics.get('labels', []))}")
            
            st.markdown("#### Confusion Matrix")
            cm = metrics.get("confusion_matrix")
            labels = metrics.get("labels", [])
            if cm and labels:
                header = "| Actual / Predicted | " + " | ".join(labels) + " |"
                separator = "|---|" + "|".join(["---"] * len(labels)) + "|"
                st.markdown(header)
                st.markdown(separator)
                for i, row in enumerate(cm):
                    row_str = " | ".join([str(x) for x in row])
                    st.markdown(f"| **{labels[i]}** | {row_str} |")
except Exception as e:
    st.error(f"Error loading model metrics: {e}")

"""
original_code = original_code.replace(
    "# ==========================================\n# DASHBOARD MODE\n# ==========================================",
    model_perf_block + "\n# ==========================================\n# DASHBOARD MODE\n# =========================================="
)

# 2. Extract map and signal HTML to functions to be injected inside Camera Monitoring section
helpers_and_camera = """if mode == "📹 Camera Monitoring":

    def get_route_color(level):
        if level == "Low": return "#22c55e"
        elif level == "Moderate": return "#f97316"
        elif level == "High": return "#ef4444"
        else: return "#7f1d1d"

    def draw_live_traffic_map(route_data):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        fig, ax = plt.subplots(figsize=(11, 5.5))
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")
        
        route_a_x = [0.3, 4.7]; route_a_y = [3, 3]
        route_b_x = [5.3, 9.7]; route_b_y = [3, 3]
        main_x = [5, 5]; main_y = [0.3, 5.7]
        
        ax.plot([0.2, 9.8], [3, 3], linewidth=30, color="#cbd5e1", solid_capstyle="round", zorder=1)
        ax.plot(main_x, main_y, linewidth=24, color="#cbd5e1", solid_capstyle="round", zorder=1)
        
        color_a = get_route_color(route_data.get("Route A", {}).get("congestion", "Unknown"))
        color_b = get_route_color(route_data.get("Route B", {}).get("congestion", "Unknown"))
        
        ax.plot(route_a_x, route_a_y, linewidth=22, color=color_a, solid_capstyle="butt", alpha=0.85, zorder=2)
        ax.plot(route_b_x, route_b_y, linewidth=22, color=color_b, solid_capstyle="butt", alpha=0.85, zorder=2)
        ax.plot(main_x, main_y, linewidth=18, color="#94a3b8", solid_capstyle="butt", alpha=0.9, zorder=2)
        
        ax.plot([0.5, 4.5], [3, 3], linewidth=1.5, color="#64748b", linestyle="--", dashes=(6, 4), alpha=0.5, zorder=3)
        ax.plot([5.5, 9.5], [3, 3], linewidth=1.5, color="#64748b", linestyle="--", dashes=(6, 4), alpha=0.5, zorder=3)
        ax.plot([5, 5], [0.5, 2.6], linewidth=1.5, color="#64748b", linestyle="--", dashes=(5, 4), alpha=0.5, zorder=3)
        ax.plot([5, 5], [3.4, 5.5], linewidth=1.5, color="#64748b", linestyle="--", dashes=(5, 4), alpha=0.5, zorder=3)
        
        ax.add_patch(plt.Circle((5, 3), 0.52, color="#e2e8f0", zorder=4))
        ax.add_patch(plt.Circle((5, 3), 0.52, fill=False, edgecolor="#fbbf24", linewidth=2.5, zorder=5))
        ax.text(5, 3, "🚦", fontsize=20, ha="center", va="center", zorder=6)
        
        ax.text(2.5, 3.65, "ROUTE A", fontsize=11, fontweight="bold", color="#1e293b", ha="center", va="center", zorder=7)
        ax.text(7.5, 3.65, "ROUTE B", fontsize=11, fontweight="bold", color="#1e293b", ha="center", va="center", zorder=7)
        
        if "Route A" in route_data:
            r = route_data["Route A"]
            ax.text(2.5, 2.05, f"🚗 {r['vehicles']} vehicles   Density: {r['density']*100:.1f}%\\n{r['congestion']}   Green: {r['green_time']}s", ha="center", va="center", fontsize=8.5, color="#334155", zorder=7, bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor=color_a, linewidth=1.5, alpha=0.9))
        if "Route B" in route_data:
            r = route_data["Route B"]
            ax.text(7.5, 2.05, f"🚗 {r['vehicles']} vehicles   Density: {r['density']*100:.1f}%\\n{r['congestion']}   Green: {r['green_time']}s", ha="center", va="center", fontsize=8.5, color="#cbd5e1", zorder=7, bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor=color_b, linewidth=1.5, alpha=0.9))
            
        ax.text(5, 5.55, "▲  NORTH", ha="center", fontsize=9, fontweight="bold", color="#94a3b8", zorder=7)
        ax.text(5, 0.45, "▼  SOUTH", ha="center", fontsize=9, fontweight="bold", color="#94a3b8", zorder=7)
        
        legend_patches = [
            mpatches.Patch(color="#22c55e", label="Low"),
            mpatches.Patch(color="#f97316", label="Moderate"),
            mpatches.Patch(color="#ef4444", label="High"),
            mpatches.Patch(color="#7f1d1d", label="Critical"),
        ]
        legend = ax.legend(handles=legend_patches, loc="upper left", frameon=True, framealpha=0.95, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8, title="Congestion", title_fontsize=8, labelcolor="#374151")
        legend.get_title().set_color("#64748b")
        
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 6.5)
        ax.axis("off")
        ax.set_title("AI-Powered Traffic Network — Live View", fontsize=13, fontweight="bold", color="#475569", pad=12, loc="left")
        fig.tight_layout(pad=1.0)
        return fig

    def generate_signal_html(route_a_green, route_b_green, yellow_time=3):
        total_cycle = route_a_green + yellow_time + route_b_green + yellow_time
        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Inter', Arial, sans-serif; background: transparent; }}
        .container {{ width: 100%; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border: 1px solid #e2e8f0; border-radius: 20px; padding: 22px 20px 18px; color: #0f172a; }}
        .ctrl-title {{ text-align: center; font-size: 13px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #64748b; margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid #e2e8f0; }}
        .roads {{ display: flex; align-items: stretch; justify-content: space-between; gap: 14px; }}
        .route {{ flex: 1; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 16px; padding: 16px 14px; text-align: center; transition: border-color 0.4s; }}
        .route-name {{ font-size: 13px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #475569; margin-bottom: 12px; }}
        .road {{ height: 56px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; transition: background 0.5s ease; background: #cbd5e1; color: #475569; }}
        .signal-box {{ width: 72px; background: #e2e8f0; border: 2px solid #cbd5e1; border-radius: 36px; padding: 14px 10px; display: flex; flex-direction: column; align-items: center; gap: 12px; flex-shrink: 0; box-shadow: 0 4px 24px rgba(0,0,0,0.12); }}
        .light {{ width: 42px; height: 42px; border-radius: 50%; background: #cbd5e1; transition: background 0.4s, box-shadow 0.4s; }}
        .active-red {{ background: #ef4444; box-shadow: 0 0 20px #ef4444, 0 0 40px rgba(239,68,68,0.4); }}
        .active-yellow {{ background: #facc15; box-shadow: 0 0 20px #facc15, 0 0 40px rgba(250,204,21,0.4); }}
        .active-green {{ background: #22c55e; box-shadow: 0 0 20px #22c55e, 0 0 40px rgba(34,197,94,0.4); }}
        .status {{ margin-top: 12px; display: inline-block; padding: 4px 14px; border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; background: #e2e8f0; color: #64748b; transition: background 0.5s, color 0.5s; }}
        .status-green  {{ background: #dcfce7; color: #16a34a; }}
        .status-yellow {{ background: #fef9c3; color: #ca8a04; }}
        .status-red    {{ background: #fee2e2; color: #dc2626; }}
        .timer {{ font-size: 36px; font-weight: 800; margin-top: 10px; font-family: 'Inter', monospace; color: #0f172a; letter-spacing: -1px; min-height: 44px; line-height: 1; transition: color 0.5s; }}
        .timer-green  {{ color: #16a34a; }}
        .timer-yellow {{ color: #ca8a04; }}
        .timer-red    {{ color: #dc2626; }}
        .info {{ margin-top: 16px; padding-top: 14px; border-top: 1px solid #e2e8f0; text-align: center; font-size: 11px; color: #64748b; letter-spacing: 0.5px; }}
        .info span {{ color: #374151; font-weight: 600; }}
        </style>
        </head>
        <body>
        <div class="container">
        <div class="ctrl-title">🚦 &nbsp; Smart Signal Controller &nbsp; 🚦</div>
        <div class="roads">
        <div class="route" id="routeBoxA">
        <div class="route-name">Route A</div>
        <div id="roadA" class="road">← ROUTE A →</div>
        <div id="statusA" class="status">WAITING</div>
        <div id="timerA" class="timer">--</div>
        </div>
        <div class="signal-box">
        <div id="red" class="light"></div><div id="yellow" class="light"></div><div id="green" class="light"></div>
        </div>
        <div class="route" id="routeBoxB">
        <div class="route-name">Route B</div>
        <div id="roadB" class="road">← ROUTE B →</div>
        <div id="statusB" class="status">WAITING</div>
        <div id="timerB" class="timer">--</div>
        </div>
        </div>
        <div class="info">AI Recommended Green Time &nbsp;|&nbsp; Route A = <span>{route_a_green}s</span> &nbsp;&nbsp;·&nbsp;&nbsp; Route B = <span>{route_b_green}s</span> &nbsp;&nbsp;·&nbsp;&nbsp; Cycle = <span>{total_cycle}s</span></div>
        </div>
        <script>
        const routeAGreen = {route_a_green};
        const routeBGreen = {route_b_green};
        const yellowTime = {yellow_time};
        const red = document.getElementById("red");
        const yellow = document.getElementById("yellow");
        const green = document.getElementById("green");
        const roadA = document.getElementById("roadA");
        const roadB = document.getElementById("roadB");
        const statusA = document.getElementById("statusA");
        const statusB = document.getElementById("statusB");
        const timerA = document.getElementById("timerA");
        const timerB = document.getElementById("timerB");
        const routeBoxA = document.getElementById("routeBoxA");
        const routeBoxB = document.getElementById("routeBoxB");
        function resetLights() {{ red.className="light"; yellow.className="light"; green.className="light"; }}
        function routeAGreenPhase(seconds) {{
            resetLights(); green.className="light active-green"; roadA.style.background="#166534"; roadB.style.background="#450a0a";
            statusA.innerHTML="🟢 GREEN"; statusA.className="status status-green"; statusB.innerHTML="🔴 STOP"; statusB.className="status status-red";
            routeBoxA.style.borderColor="#22c55e"; routeBoxB.style.borderColor="#ef4444";
            let remaining=seconds; timerA.innerHTML=remaining+"s"; timerA.className="timer timer-green"; timerB.innerHTML="WAIT"; timerB.className="timer timer-red";
            const interval=setInterval(() => {{ remaining--; if(remaining<=0) {{ clearInterval(interval); routeAYellowPhase(); }} else {{ timerA.innerHTML=remaining+"s"; }} }}, 1000);
        }}
        function routeAYellowPhase() {{
            resetLights(); yellow.className="light active-yellow"; roadA.style.background="#713f12"; roadB.style.background="#450a0a";
            statusA.innerHTML="🟡 CHANGE"; statusA.className="status status-yellow"; statusB.innerHTML="🔴 STOP"; statusB.className="status status-red";
            routeBoxA.style.borderColor="#facc15"; let remaining=yellowTime; timerA.innerHTML=remaining+"s"; timerA.className="timer timer-yellow";
            const interval=setInterval(() => {{ remaining--; if(remaining<=0) {{ clearInterval(interval); routeBGreenPhase(routeBGreen); }} else {{ timerA.innerHTML=remaining+"s"; }} }}, 1000);
        }}
        function routeBGreenPhase(seconds) {{
            resetLights(); green.className="light active-green"; roadA.style.background="#450a0a"; roadB.style.background="#166534";
            statusA.innerHTML="🔴 STOP"; statusA.className="status status-red"; statusB.innerHTML="🟢 GREEN"; statusB.className="status status-green";
            routeBoxA.style.borderColor="#ef4444"; routeBoxB.style.borderColor="#22c55e";
            let remaining=seconds; timerA.innerHTML="WAIT"; timerA.className="timer timer-red"; timerB.innerHTML=remaining+"s"; timerB.className="timer timer-green";
            const interval=setInterval(() => {{ remaining--; if(remaining<=0) {{ clearInterval(interval); routeBYellowPhase(); }} else {{ timerB.innerHTML=remaining+"s"; }} }}, 1000);
        }}
        function routeBYellowPhase() {{
            resetLights(); yellow.className="light active-yellow"; roadA.style.background="#450a0a"; roadB.style.background="#713f12";
            statusA.innerHTML="🔴 STOP"; statusA.className="status status-red"; statusB.innerHTML="🟡 CHANGE"; statusB.className="status status-yellow";
            routeBoxB.style.borderColor="#facc15"; let remaining=yellowTime; timerB.innerHTML=remaining+"s"; timerB.className="timer timer-yellow";
            const interval=setInterval(() => {{ remaining--; if(remaining<=0) {{ clearInterval(interval); routeAGreenPhase(routeAGreen); }} else {{ timerB.innerHTML=remaining+"s"; }} }}, 1000);
        }}
        routeAGreenPhase(routeAGreen);
        </script>
        </body>
        </html>
        '''
        return html

    st.divider()

    st.markdown('''
    <div class="dash-title" style="font-size:24px;">🎥 AI Live Traffic Monitoring</div>
    <div style="font-size:13px; font-weight:700; color:#dc2626; letter-spacing:1px; margin-bottom:16px;">● LIVE ANALYSIS READY</div>
    ''', unsafe_allow_html=True)

    up_col1, up_col2 = st.columns(2)
    
    with up_col1:
        st.markdown("### 🛣️ Route A Window")
        route_a_file = st.file_uploader("Upload Route A Video", type=["mp4", "avi", "mov", "mkv"], key="upload_a")
    
    with up_col2:
        st.markdown("### 🛣️ Route B Window")
        route_b_file = st.file_uploader("Upload Route B Video", type=["mp4", "avi", "mov", "mkv"], key="upload_b")

    if route_a_file and route_b_file:
        
        os.makedirs("videos", exist_ok=True)
        temp_a_path = os.path.join("videos", "upload_route_a_temp.mp4")
        temp_b_path = os.path.join("videos", "upload_route_b_temp.mp4")
        
        if st.button("▶ Start Live AI Analysis"):
            with open(temp_a_path, "wb") as f:
                f.write(route_a_file.read())
            with open(temp_b_path, "wb") as f:
                f.write(route_b_file.read())
            
            st.session_state.live_analysis_running_v2 = True
            
        if not st.session_state.get("live_analysis_running_v2", False):
            st.divider()
            video_col1, video_col2 = st.columns(2)
            with video_col1:
                st.video(route_a_file)
            with video_col2:
                st.video(route_b_file)

        if st.session_state.get("live_analysis_running_v2", False):
            
            st.divider()
            
            status_container = st.empty()
            videos_container = st.empty()
            visualizer_container = st.empty()
            
            try:
                fps_a, total_frames_a, duration_a = get_video_info(temp_a_path)
                fps_b, total_frames_b, duration_b = get_video_info(temp_b_path)
                
                max_duration = max(duration_a, duration_b)
                current_sec = 0
                
                while current_sec < max_duration:
                    end_sec = current_sec + 5
                    progress_ratio = min(1.0, current_sec / max_duration)
                    
                    with status_container.container():
                        st.markdown(f"**ANALYSIS STATUS**")
                        st.markdown(f"Currently analyzing: {current_sec}–{end_sec} seconds")
                        st.progress(progress_ratio)
                    
                    results_a = analyze_video_window(temp_a_path, current_sec, 5)
                    results_b = analyze_video_window(temp_b_path, current_sec, 5)
                    
                    if not results_a and not results_b:
                        break
                        
                    if not results_a: results_a = {"Cars": 0, "Motorcycles": 0, "Buses": 0, "Trucks": 0, "Total_Vehicles": 0}
                    if not results_b: results_b = {"Cars": 0, "Motorcycles": 0, "Buses": 0, "Trucks": 0, "Total_Vehicles": 0}
                        
                    pred_a, total_a, den_a = predict_congestion(model, results_a["Cars"], results_a["Motorcycles"], results_a["Buses"], results_a["Trucks"])
                    pred_b, total_b, den_b = predict_congestion(model, results_b["Cars"], results_b["Motorcycles"], results_b["Buses"], results_b["Trucks"])
                    
                    cong_a = congestion_label(pred_a)
                    cong_b = congestion_label(pred_b)
                    
                    signal_data = optimize_signals({"Total_Vehicles": total_a}, {"Total_Vehicles": total_b})
                    
                    green_a = signal_data["Route A"]["Green_Time"]
                    green_b = signal_data["Route B"]["Green_Time"]
                    
                    # Create route data exactly how Simulation code expects it for drawing the plot
                    route_data_map = {
                        "Route A": {"vehicles": total_a, "density": den_a, "congestion": cong_a, "green_time": green_a},
                        "Route B": {"vehicles": total_b, "density": den_b, "congestion": cong_b, "green_time": green_b}
                    }
                    
                    fig = draw_live_traffic_map(route_data_map)
                    html_code = generate_signal_html(green_a, green_b, 3)
                    
                    with videos_container.container():
                        v_col1, v_col2 = st.columns(2)
                        with v_col1:
                            st.markdown(f"**Route A `({current_sec}-{end_sec}s)`**")
                            st.video(temp_a_path, start_time=current_sec)
                        with v_col2:
                            st.markdown(f"**Route B `({current_sec}-{end_sec}s)`**")
                            st.video(temp_b_path, start_time=current_sec)
                            
                    with visualizer_container.container():
                        st.markdown("---")
                        # Emulate layout: Map on the left, Side-by-side stats below it or left/right
                        top_left, top_right = st.columns([1, 1.6], gap="large")
                        
                        with top_right:
                            st.pyplot(fig, clear_figure=True)
                            
                        with top_left:
                            st.markdown('<div class="sim-section-title">🛣️ ROUTE A LIVE</div>', unsafe_allow_html=True)
                            st.markdown(f"**Cars:** {total_a}")
                            st.markdown(f"**Congestion:** {cong_a}")
                            st.markdown(f"**Density:** {den_a*100:.1f}%")
                            st.markdown(f"**Green:** {green_a}s")
                            
                            st.markdown('<div class="sim-section-title" style="margin-top:16px;">🛣️ ROUTE B LIVE</div>', unsafe_allow_html=True)
                            st.markdown(f"**Cars:** {total_b}")
                            st.markdown(f"**Congestion:** {cong_b}")
                            st.markdown(f"**Density:** {den_b*100:.1f}%")
                            st.markdown(f"**Green:** {green_b}s")
                        
                        st.markdown("---")
                        
                        bot_left, bot_right = st.columns([1.5, 1], gap="large")
                        with bot_left:
                            components.html(html_code, height=440, scrolling=False)
                                
                    current_sec += 5
                    
                status_container.markdown("**Status:** `● Analysis Complete`")
                st.session_state.live_analysis_running_v2 = False
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.session_state.live_analysis_running_v2 = False

"""

# Use string manipulation to replace the old camera mode logic.
start_idx = original_code.find('if mode == "📹 Camera Monitoring":')
end_idx = original_code.find('elif mode == "🎮 Simulation Dashboard":')

if start_idx != -1 and end_idx != -1:
    new_app = original_code[:start_idx] + helpers_and_camera + "\n\n" + original_code[end_idx:]
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(new_app)
    print("SUCCESS: Camera Logic Updated")
else:
    print("ERROR: Could not find block indices.")
