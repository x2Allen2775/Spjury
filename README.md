# ⚡ SPJURY — AI Powered Sports Injury Predictor
### Presented by Team SPJURY

A cyberpunk-styled web application for real-time biomechanics analysis and injury prediction in Cricket (Batting & Bowling) and Tennis using computer vision and MediaPipe pose estimation.

---

## 🚀 QUICK START

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
python app.py
```

### 3. Open in Browser
```
http://localhost:5000
```

---

## 📁 PROJECT STRUCTURE

```
spjury_app/
├── app.py                        ← Flask backend
├── requirements.txt
├── templates/
│   └── index.html                ← Cyberpunk UI frontend
├── analyzers/
│   ├── __init__.py
│   ├── bowling_module.py         ← Cricket bowling analysis (adapted)
│   ├── batting_module.py         ← Cricket batting analysis (adapted)
│   └── tennis_module.py          ← Tennis analysis (adapted)
├── uploads/                      ← Temp video uploads (auto-cleaned)
└── outputs/                      ← Analysis results per task
```

---

## 🏏 BOWLING ANALYSIS FEATURES
- 8-Phase detection (Run-Up → Gather → Jump → Plant → Delivery → Release → Follow-Through)
- **ICC Legality Check** — 15° elbow extension rule (LEGAL ✅ / ILLEGAL 🚫)
- Phase-by-Phase injury risk scoring
- Action classification: Front-On / Side-On / Mixed
- Bowling type detection: Fast / Medium / Spin
- Per-delivery comparison charts
- Injury types: Lumbar Disc, Rotator Cuff, Labral Tear, Patellar Tendinopathy, etc.

## 🏏 BATTING ANALYSIS FEATURES
- Auto-detection: Batting vs Bowling
- Weight transfer (front foot / back foot %)
- Knee and Hip angle tracking
- Injury risk: Knee Joint Stress, Lower Back Stress, Weight Imbalance
- Real-time risk timeline visualization

## 🎾 TENNIS ANALYSIS FEATURES
- Shoulder abduction angle
- Elbow flexion & angular velocity
- Knee flexion monitoring
- Trunk lateral tilt (lower back risk)
- Hip-shoulder separation analysis
- Multi-factor injury risk scoring
- In-video real-time graph overlays

---

## 📊 OUTPUT FILES (per analysis)
| File | Description |
|------|-------------|
| `annotated_*.mp4` | Video with pose overlays & annotations |
| `*.png` graphs | Biomechanics visualization charts |
| `*_analysis.json` | Full structured data |
| `*.csv` | Summary tabular data |

---

## ⚠ NOTES
- Large videos may take several minutes to process
- MediaPipe requires Python 3.8–3.11
- GPU acceleration is not required but helps with larger videos
- Uploaded videos are deleted after processing for privacy

---

## 🛡 ICC LEGALITY CHECK (Bowling)
Per ICC regulations, a bowler's elbow must not extend by more than **15°** during delivery.
- ✅ **LEGAL**: Extension ≤ 15°
- 🚫 **ILLEGAL**: Extension > 15° (chucking)

This is determined per delivery and displayed prominently in the results.


## AI Chat Feature
The AI Coach chat panel uses **Groq** (free — no credit card needed).  
Get your free API key at **console.groq.com** → API Keys.

Set it as an environment variable **before** starting the app:

```bash
# Linux / macOS
export GROQ_API_KEY=gsk_...

# Windows (PowerShell)
$env:GROQ_API_KEY="gsk_..."

# Then start the app
python app.py
```

The key is never exposed to the browser. All chat requests are proxied through the Flask server at `/api/chat`.  
If the key is not set, the chat panels will show a warning and remain disabled.
