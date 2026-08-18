<div align="center">
<a href="https://ai-intelligent-traffic-analyzer.streamlit.app/">🚀 LIVE DEMO — Open Traffic Intelligence Dashboard</a>


# 🚦 AI TRAFFIC INTELLIGENCE

### `AI-Powered Traffic Monitoring • Congestion Prediction • Adaptive Signal Control`

<img src="https://img.shields.io/badge/AI-Traffic%20Intelligence-00D4FF?style=for-the-badge&logo=artificial-intelligence" />
<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/YOLO-Computer%20Vision-111111?style=for-the-badge" />
<img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />

<br><br>

<a href="https://ai-intelligent-traffic-analyzer.streamlit.app/">
  <img src="https://img.shields.io/badge/🚀%20LIVE%20DEMO-Open%20Dashboard-00C853?style=for-the-badge" />
</a>

<br><br>

**Turning traffic camera footage into intelligent, adaptive traffic decisions.**

<br>

`VIDEO INPUT` → `VEHICLE DETECTION` → `TRAFFIC INTELLIGENCE` → `ML PREDICTION` → `SMART SIGNAL`

</div>

---

<div align="center">

## 🧠 THE SYSTEM

</div>


<table>
<tr>

<td width="50%" valign="top">

### 🎥 LIVE TRAFFIC MONITORING

Upload traffic footage from:

* 🛣️ Route A
* 🛣️ Route B

The system processes the videos in progressive time windows:

```text
00:00 ──► 00:05 ──► 00:10 ──► 00:15 ──► ...
```

Each completed window updates the traffic intelligence dashboard.

</td>

<td width="50%" valign="top">

### 🚦 ADAPTIVE SIGNAL CONTROL

Traffic conditions are converted into AI signal recommendations.

```text
Traffic
   ↓
Congestion
   ↓
Prediction
   ↓
Green Time
   ↓
Signal Controller
```

The next signal cycle can use the latest prediction without unnecessarily interrupting the active cycle.

</td>

</tr>
</table>

---

<div align="center">

## ⚡ HOW IT WORKS

</div>

```text
╔══════════════════════════════════════════════════════════════╗
║                    TRAFFIC CAMERA INPUT                     ║
║                    Route A + Route B                        ║
╚══════════════════════════════════════════════════════════════╝
                              │
                              ▼
                    ┌──────────────────┐
                    │   YOLO DETECTOR  │
                    │                  │
                    │ 🚗 Cars          │
                    │ 🏍️ Motorcycles   │
                    │ 🚌 Buses         │
                    │ 🚛 Trucks        │
                    └────────┬─────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ TRAFFIC INTELLIGENCE │
                  │                      │
                  │ Vehicle Count        │
                  │ Density              │
                  │ Traffic Share        │
                  └──────────┬───────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  ML PREDICTION  │
                    │                │
                    │ Low            │
                    │ Moderate       │
                    │ High           │
                    │ Critical       │
                    └───────┬────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  SMART SIGNAL AI    │
                 │                      │
                 │ Route A → Green Time │
                 │ Route B → Green Time │
                 └──────────┬───────────┘
                            │
                            ▼
                 ╔══════════════════════╗
                 ║  LIVE DASHBOARD      ║
                 ║  MAP + SIGNALS      ║
                 ║  TRAFFIC + AI DATA  ║
                 ╚══════════════════════╝
```

---

<div align="center">

## 📊 AI MODEL PERFORMANCE

### Current Evaluation Results

<table>
<tr>
<td align="center">

### 🎯 Accuracy

# 90.91%

</td>

<td align="center">

### 🎯 Precision

# 92.05%

</td>

<td align="center">

### 🔄 Recall

# 90.91%

</td>

<td align="center">

### ⚡ F1 Score

# 89.70%

</td>
</tr>
</table>

</div>

> These results are based on the current evaluation dataset and should not be interpreted as a universal accuracy guarantee for unseen traffic environments.

---

<div align="center">

## 🚦 CONGESTION INTELLIGENCE

</div>

| Level | Meaning      | System Interpretation            |
| :---: | ------------ | -------------------------------- |
|   🟢  | **Low**      | Traffic flow is relatively light |
|   🟡  | **Moderate** | Normal/moderate traffic pressure |
|   🟠  | **High**     | Significant traffic congestion   |
|   🔴  | **Critical** | Severe congestion                |

---

<div align="center">

## 🗺️ REAL-TIME TRAFFIC SIMULATION

</div>

The simulation dashboard visualizes the traffic network and demonstrates how changing traffic conditions influence the intelligent signal controller.

```text
                    🚦 INTERSECTION
                         │
             ┌───────────┴───────────┐
             │                       │
        🛣️ ROUTE A               🛣️ ROUTE B
             │                       │
       🚗 🚗 🚗 🚗              🚗 🚗
       🚗 🚗 🚗                🚗
             │                       │
             └───────────┬───────────┘
                         │
                    🤖 AI CONTROL
                         │
              ┌──────────┴──────────┐
              │                     │
         🟢 GREEN TIME         🟢 GREEN TIME
            Route A                Route B
```

---

<div align="center">

## 📹 LIVE VIDEO ANALYSIS

</div>

### Progressive Analysis

```text
┌───────────────────────────────────────────┐
│ 🔄 ANALYZING ROUTE A + ROUTE B            │
├───────────────────────────────────────────┤
│                                           │
│  00 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 05 sec   │
│                                           │
│  🚗 Cars              1085                │
│  🚌 Buses              91                │
│  🚛 Trucks            253                │
│  🚦 Congestion        HIGH                │
│                                           │
│  🤖 AI GREEN TIME                        │
│                                           │
│  Route A                 40 sec            │
│  Route B                 60 sec            │
│                                           │
└───────────────────────────────────────────┘
```

Then the next window is processed:

```text
05 sec ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10 sec
```

and the dashboard updates with the latest traffic prediction.

---

<div align="center">

## 🧩 TECHNOLOGY STACK

</div>

<table>
<tr>
<td align="center">🐍<br><b>Python</b></td>
<td align="center">👁️<br><b>YOLO</b></td>
<td align="center">🎞️<br><b>OpenCV</b></td>
<td align="center">🧠<br><b>Scikit-learn</b></td>
<td align="center">📊<br><b>Streamlit</b></td>
</tr>
</table>

---

## 📁 PROJECT ARCHITECTURE

```text
AI-Intelligent-Traffic-Analyzer/
│
├── 🚦 app.py
│
├── 🤖 live_traffic_engine.py
├── 🎥 video_live_analyzer.py
├── 🚗 vehicle_detection.py
├── 🛰️ vehicle_tracking.py
│
├── 🧠 traffic_intelligence.py
├── 📈 traffic_prediction.py
├── 🗺️ traffic_simulation.py
├── 🚦 signal_optimizer.py
├── 🛣️ route_analysis.py
│
├── 🧪 model_evaluation.py
├── 🏋️ train_model.py
├── 🧹 prepare_dataset.py
├── 🎞️ process_videos.py
│
├── 📦 traffic_model.pkl
├── 📊 model_metrics.pkl
├── 🤖 yolo11n.pt
│
├── 📂 data/
├── 📂 videos/
└── 📂 processed_videos/
```

---

<div align="center">

## 🔬 MODEL EVALUATION

</div>

Run:

```bash
python model_evaluation.py
```

The evaluation generates:

```text
Accuracy
Precision
Recall
F1 Score
Classification Report
Confusion Matrix
```

The resulting metrics are saved to:

```text
model_metrics.pkl
```

---

<details>
<summary><b>📊 View Classification Performance</b></summary>

<br>

| Class    | Precision | Recall |   F1 |
| -------- | --------: | -----: | ---: |
| Low      |      1.00 |   0.50 | 0.67 |
| Moderate |      0.88 |   1.00 | 0.93 |
| High     |      1.00 |   1.00 | 1.00 |
| Critical |      0.00 |   0.00 | 0.00 |

The current evaluation dataset contains no Critical samples, so the Critical class has zero support in this evaluation.

</details>

---

<details>
<summary><b>🚀 Run Locally</b></summary>

<br>

```bash
git clone <YOUR-REPOSITORY-URL>

cd AI-Intelligent-Traffic-Analyzer

python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start dashboard

```bash
streamlit run app.py
```

</details>

---

<div align="center">

## 🔮 FUTURE ROADMAP

```text
        CURRENT
           │
           ▼
   🎥 Video Analysis
           │
           ▼
   🤖 ML Prediction
           │
           ▼
   🚦 Adaptive Signals
           │
           ▼
   🌐 Cloud Deployment
           │
           ▼
   📡 Real CCTV Streams
           │
           ▼
   🏙️ Multi-Intersection AI
           │
           ▼
   🚑 Emergency Vehicle Priority
```

</div>

---

<div align="center">

## 🎯 VISION

### `SEE → UNDERSTAND → PREDICT → OPTIMIZE`

**The goal is simple:**

> Build an intelligent traffic-management system that can understand real-world traffic conditions and continuously optimize traffic flow.

<br>

### 🚦 AI TRAFFIC INTELLIGENCE

`Computer Vision` × `Machine Learning` × `Traffic Intelligence` × `Adaptive Control`

<br><br>

⭐ **Star this repository if you like the project.**

</div>
