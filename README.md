# 🧠 CogniLearn Enterprise — AI-Based Cognitive Learning Pattern Analyzer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)

**Cognitive Learning Pattern Analyzer** is an AI-powered platform that analyzes student learning behavior — response time, retry patterns, and mistake frequency — to model cognitive patterns and deliver personalized adaptive learning strategies.

---

## 🎯 Problem Statement

**Challenge:** Traditional education lacks tools to analyze *how* students learn, not just *what* they score. Response time, error patterns, and retry behavior contain rich signals about cognitive style that go unanalyzed.

**Solution:** This system captures behavioral data from every quiz attempt, extracts 8+ behavioral features, classifies students into one of 5 cognitive patterns using ML, and generates fully personalized adaptive recommendations — all without requiring an API key.

---

## ✨ Must-Have Features (Problem Statement Requirements)

### 1. 🔬 Behavioral Data Analysis
Extracted from every quiz attempt:
- `avg_response_time_s` — time per question (confidence & processing speed)
- `mistake_frequency` — proportion of wrong answers across attempts
- `avg_retry_rate` — how often students repeat quizzes
- `speed_accuracy_ratio` — combined speed + correctness index
- `topic_consistency` — uniformity of performance across topics
- `improvement_slope` — regression slope of scores over time
- `hard_difficulty_bias` — preference/avoidance of hard questions

### 2. 🧠 Learning Pattern Classification
Five cognitive patterns classified by rule-based ML:

| Pattern | Description |
|---|---|
| 🚀 **Rapid Master** | Fast + accurate + consistent — high confidence |
| 🔬 **Analytical Thinker** | Slow but accurate — deliberate reasoning |
| ⚡ **Impulsive Responder** | Fast but error-prone — needs to slow down |
| 📚 **Foundational Learner** | Low accuracy + no improvement — needs structured review |
| 🔄 **Strategic Revisor** | High retry + improving — active mistake learner |

### 3. 🎯 Adaptive Recommendation Engine
Fully local (no API key required). Generates 4–6 personalized recommendations:
- Pattern-based study strategy (e.g., Spaced Repetition for Foundational Learner)
- Response-time coaching (too slow → concept gaps; too fast → impulsive errors)
- Retry habit analysis (high retries → direct to first-attempt improvement)
- Score trend intervention (declining → fatigue alert; improving → maintain momentum)
- Topic inconsistency fix (identifies specific weak topics to target)
- Course progress boost (pairs quizzing with lesson completion)

Each recommendation shows **behavioral evidence** (e.g., "Accuracy: 62% | Avg time: 95s/q").

### 4. 📊 Performance Tracking Dashboard
- 6 KPI metrics (progress, score, XP, quizzes)
- Course completion bar chart
- Aptitude radar (Memory, Analysis, Focus, Speed, Application)
- Behavioral features table with all 8 extracted metrics
- Behavioral profile radar (Accuracy, Speed, Consistency, First-Try Success, Improvement)

### 5. 📈 Improvement Analytics Report
- Average, best, and worst scores
- Week-over-week score change
- **Predicted next score** (linear extrapolation)
- Score trajectory chart with trend line + prediction marker
- Achievement milestones (Excellence, Committed Learner, Rising Star…)
- Behavioral strengths & improvement areas
- Strongest and weakest topics across all quizzes

---

## 📂 Repository Structure

```text
cognitive_learning/
│
├── README.md               ← Project documentation
├── requirements.txt        ← Python dependencies
│
├── config/
│   ├── __init__.py
│   └── data.py             ← Course syllabus, user DB, difficulty & UI data
│
├── core/
│   ├── __init__.py
│   └── engine.py           ← NLP extraction, Claude API, quiz generation,
│                              + BEHAVIORAL ANALYSIS ENGINE (new)
│                              · compute_behavioral_features()
│                              · classify_cognitive_pattern()
│                              · generate_adaptive_recommendations()
│                              · generate_improvement_report()
│
├── ui/
│   ├── __init__.py
│   └── components.py       ← All Streamlit renderers including:
│                              · render_behavioral_analytics()
│                              · render_adaptive_recommendations()
│                              · render_improvement_report()
│                              · render_pattern_card()
│
└── app.py                  ← Main entry point with enhanced Analytics tab
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/your-username/cognitive-learning.git
cd cognitive_learning
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Demo credentials:** Registration Number `43611162` / Password `pass`

---

## 🔑 Optional: Claude AI Integration

The platform works fully **offline** — no API key required. To unlock Claude-powered quiz generation and AI recommendations:

1. Obtain an API key from [console.anthropic.com](https://console.anthropic.com).
2. Enter it in the **sidebar** under *🔑 Anthropic API Key*.

---

## 🧩 Module Responsibilities

| Module | Responsibility |
|---|---|
| `config/data.py` | Static data: stopwords, difficulty, 6-course syllabus, brain exercises, school lists |
| `core/engine.py` | Text extraction (PDF/DOCX/PPTX/TXT), local NLP, Claude API, **behavioral analysis engine** |
| `ui/components.py` | Theme, CSS injection, quiz/exam/results renderers, **behavioral analytics UI**, pattern card |
| `app.py` | Auth, sidebar with live pattern badge, Student Dashboard, AI Analytics Portal |

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `plotly` | Interactive charts |
| `scikit-learn` | KMeans, PCA, StandardScaler, classifiers |
| `pandas / numpy` | Data manipulation |
| `networkx / matplotlib` | Feature interaction network |
| `pdfplumber` | PDF extraction |
| `python-docx` | DOCX extraction |
| `python-pptx` | PPTX extraction |
| `requests` | Claude API calls |

---

## 📄 License

MIT License — see `LICENSE` for details.
