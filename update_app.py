import sys
import os

app_path = r'c:\Users\Shivangi\OneDrive\Desktop\Rohan\Projects\AI-Traffic-Intelligence\app.py'

with open(app_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_imports = "import joblib\nimport time\nimport os\nfrom live_traffic_engine import analyze_video_window, get_video_info\n"

# Replace import joblib
for i, line in enumerate(lines):
    if line.strip() == "import joblib":
        lines[i] = new_imports
        break

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if line.startswith('if mode == "📹 Camera Monitoring":'):
        start_idx = i
    elif line.startswith('elif mode == "🎮 Simulation Dashboard":'):
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    
    new_block = """if mode == "📹 Camera Monitoring":

    st.divider()

    st.markdown('''
    <div class="dash-title" style="font-size:24px;">🎥 AI Live Traffic Monitoring</div>
    <div style="font-size:13px; font-weight:700; color:#dc2626; letter-spacing:1px; margin-bottom:16px;">● LIVE ANALYSIS READY</div>
    ''', unsafe_allow_html=True)

    up_col1, up_col2 = st.columns(2)
    
    with up_col1:
        st.markdown("### 🛣️ Route A")
        route_a_file = st.file_uploader("Upload Route A Video", type=["mp4", "avi", "mov", "mkv"], key="upload_a")
    
    with up_col2:
        st.markdown("### 🛣️ Route B")
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
            
            st.session_state.live_analysis_running = True
            
        if not st.session_state.get("live_analysis_running", False):
            st.divider()
            video_col1, video_col2 = st.columns(2)
            with video_col1:
                st.video(route_a_file)
            with video_col2:
                st.video(route_b_file)

        if st.session_state.get("live_analysis_running", False):
            
            st.divider()
            
            status_container = st.empty()
            videos_container = st.empty()
            analysis_container = st.empty()
            details_container = st.empty()
            
            try:
                fps_a, total_frames_a, duration_a = get_video_info(temp_a_path)
                fps_b, total_frames_b, duration_b = get_video_info(temp_b_path)
                
                max_duration = max(duration_a, duration_b)
                current_sec = 0
                
                while current_sec < max_duration:
                    end_sec = current_sec + 5
                    
                    status_container.markdown(f"**Current Analysis:** `Analyzing {current_sec}–{end_sec} seconds...`")
                    
                    results_a = analyze_video_window(temp_a_path, current_sec, 5)
                    results_b = analyze_video_window(temp_b_path, current_sec, 5)
                    
                    if not results_a and not results_b:
                        break
                        
                    if not results_a:
                        results_a = {"Cars": 0, "Motorcycles": 0, "Buses": 0, "Trucks": 0, "Total_Vehicles": 0}
                    if not results_b:
                        results_b = {"Cars": 0, "Motorcycles": 0, "Buses": 0, "Trucks": 0, "Total_Vehicles": 0}
                        
                    pred_a, total_a, den_a = predict_congestion(model, results_a["Cars"], results_a["Motorcycles"], results_a["Buses"], results_a["Trucks"])
                    pred_b, total_b, den_b = predict_congestion(model, results_b["Cars"], results_b["Motorcycles"], results_b["Buses"], results_b["Trucks"])
                    
                    cong_a = congestion_label(pred_a)
                    cong_b = congestion_label(pred_b)
                    
                    signal_data = optimize_signals({"Total_Vehicles": total_a}, {"Total_Vehicles": total_b})
                    
                    green_a = signal_data["Route A"]["Green_Time"]
                    green_b = signal_data["Route B"]["Green_Time"]
                    share_a = signal_data["Route A"]["Traffic_Share"]
                    share_b = signal_data["Route B"]["Traffic_Share"]
                    
                    with videos_container.container():
                        v_col1, v_col2 = st.columns(2)
                        with v_col1:
                            st.markdown("**🛣️ Route A**")
                            st.video(temp_a_path, start_time=current_sec)
                        with v_col2:
                            st.markdown("**🛣️ Route B**")
                            st.video(temp_b_path, start_time=current_sec)
                            
                    with analysis_container.container():
                        st.markdown(f"### Current Analysis (`{current_sec}–{end_sec} seconds`)")
                        ac1, ac2, ac3 = st.columns(3)
                        
                        with ac1:
                            st.markdown("#### 🛣️ Route A")
                            st.markdown(f"- **Vehicles:** {total_a}")
                            st.markdown(f"- **Congestion:** {cong_a}")
                            
                        with ac2:
                            st.markdown("#### 🛣️ Route B")
                            st.markdown(f"- **Vehicles:** {total_b}")
                            st.markdown(f"- **Congestion:** {cong_b}")
                            
                        with ac3:
                            st.markdown("#### 🚦 AI Signal Control")
                            sig_a, sig_b = ("🟢 GREEN", "🔴 RED") if green_a >= green_b else ("🔴 RED", "🟢 GREEN")
                            st.markdown(f'''| Route | Signal | Green Time |
|---|---|---|
| **Route A** | {sig_a} | {green_a} sec |
| **Route B** | {sig_b} | {green_b} sec |''')
                            
                    with details_container.container():
                        with st.expander("🔍 Show Analysis Details"):
                            dc1, dc2 = st.columns(2)
                            with dc1:
                                st.markdown("##### Route A")
                                st.write(f"- Cars: {results_a['Cars']}")
                                st.write(f"- Motorcycles: {results_a['Motorcycles']}")
                                st.write(f"- Buses: {results_a['Buses']}")
                                st.write(f"- Trucks: {results_a['Trucks']}")
                                st.write(f"- Total Vehicles: {total_a}")
                                st.write(f"- Density: {den_a:.2f}")
                                st.write(f"- Congestion: {cong_a}")
                                st.write(f"- Traffic Share: {share_a*100:.1f}%")
                                st.write(f"- AI Green Time: {green_a}s")
                            with dc2:
                                st.markdown("##### Route B")
                                st.write(f"- Cars: {results_b['Cars']}")
                                st.write(f"- Motorcycles: {results_b['Motorcycles']}")
                                st.write(f"- Buses: {results_b['Buses']}")
                                st.write(f"- Trucks: {results_b['Trucks']}")
                                st.write(f"- Total Vehicles: {total_b}")
                                st.write(f"- Density: {den_b:.2f}")
                                st.write(f"- Congestion: {cong_b}")
                                st.write(f"- Traffic Share: {share_b*100:.1f}%")
                                st.write(f"- AI Green Time: {green_b}s")
                            st.write(f"**Processing Status:** Working window {current_sec}-{end_sec}s")
                                
                    current_sec += 5
                    
                status_container.markdown("**Status:** `● Analysis Complete`")
                st.session_state.live_analysis_running = False
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.session_state.live_analysis_running = False

"""
    lines[start_idx:end_idx] = [new_block + "\n"]

    with open(app_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("Optimization Complete")
else:
    print(f"Error: Could not find block indexes. start={start_idx}, end={end_idx}")

