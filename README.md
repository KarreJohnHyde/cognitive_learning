# 🎓 CogniLearn Enterprise — Pattern Analyzer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)

**Pattern Analyzer** is an advanced, AI-powered learning, assessment, and behavioural analytics platform. It provides a comprehensive student dashboard for tracking course progress, a **Brain Training** module that generates custom quizzes and model exams from uploaded study materials, and a dedicated **Machine Learning Analytics Portal** for exploring student performance data and cognitive patterns.

---

## 🎯 Project Problem Statement

**The Challenge:** Traditional education relies heavily on static materials (PDFs, presentations, lecture notes) which require high manual effort from students to convert into active revision practices. Furthermore, educators and administrators often lack accessible, data-driven tools to identify individual learning gaps, analyse cognitive patterns, and predict student outcomes at scale.

**The Solution:** Pattern Analyzer bridges this gap by introducing an end-to-end intelligent ecosystem. It automatically transforms static documents into dynamic, gamified assessments using NLP and Large Language Models. Simultaneously, it provides an integrated Machine Learning environment that clusters student behaviour and predicts academic performance, enabling highly personalised, adaptive learning interventions based on recognised data patterns.

---

## ✨ Key Features

- **🎓 Smart Student Dashboard:** Track active courses, syllabus progress, and daily learning streaks with gamified XP.
- **🧠 AI Assessment Generator:** Upload PDF, DOCX, PPTX, or TXT study materials to automatically extract knowledge, summarise concepts, and generate assessments.
  - **Local NLP Mode:** Runs entirely offline — no API key needed.
  - **Claude AI Integration:** *(Optional)* Input an Anthropic API key for advanced, adaptive exam generation and personalised study recommendations.
- **📊 Advanced Analytics:** Deep-dive into quiz scores, attempt frequencies, and time spent using interactive Plotly charts.
- **🤖 ML Research Portal:** Upload a student dataset (or synthesise 5,000+ records instantly) to perform EDA, 3D PCA clustering, and predictive modelling (Logistic Regression, Random Forest) to uncover hidden cognitive patterns.

---

## 📂 Repository Structure

```text
cognitive_learning/
│
├── README.md               ← Project documentation (this file)
├── requirements.txt        ← Python dependencies
│
├── config/
│   ├── __init__.py
│   └── data.py             ← Course syllabus, user DB, difficulty & UI data
│
├── core/
│   ├── __init__.py
│   └── engine.py           ← NLP text extraction, local quiz generation,
│                              Claude API helpers, AI quiz/exam/recommendation
│
├── ui/
│   ├── __init__.py
│   └── components.py       ← CSS injection, theme builder, and all Streamlit
│                              render helpers (quiz, exam paper, results, course)
│
└── app.py                  ← Main entry point: session state, authentication,
                               sidebar, Student Dashboard & AI Analytics Portal
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

The platform works fully offline with its local NLP engine. To unlock Claude-powered quiz generation, intelligent exam papers, and personalised study recommendations:

1. Obtain an API key from [console.anthropic.com](https://console.anthropic.com).
2. Enter it in the **sidebar** under *🔑 Anthropic API Key* while the app is running.

The key is never persisted — it lives only in your Streamlit session.

---

## 🧩 Module Responsibilities

| Module | Responsibility |
|---|---|
| `config/data.py` | All static data: stopwords, difficulty presets, 6-course syllabus (60+ lessons), brain-exercise definitions, school/programme lists, demo user DB |
| `core/engine.py` | Text extraction (PDF/DOCX/PPTX/TXT), local NLP pipeline (tokenise, keyword extraction, definition extraction, summarisation), Claude API wrapper, local & AI-powered quiz/exam generation, adaptive recommendations |
| `ui/components.py` | Theme builder (`build_theme`), CSS injection (`inject_css`), and five render helpers: `render_summary`, `render_quiz` (with live JS countdown), `render_exam_paper` (university-format), `render_quiz_results` (with AI recommendations), `render_course_workspace` (Coursera-style syllabus) |
| `app.py` | Streamlit page config, session-state bootstrap, event/time-tracking helpers, streak logic, authentication (sign-in + registration), sidebar, full Student Dashboard (Hub · Courses · Brain Training · Analytics), full AI Analytics Portal (Data Matrix · EDA · 3D Clusters · ML Engine) |

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `plotly` | Interactive charts (bar, scatter, 3D, heatmap, radar) |
| `scikit-learn` | KMeans clustering, PCA, StandardScaler, classification/regression |
| `pandas / numpy` | Data manipulation and numerical computing |
| `networkx` | Feature interaction network graph |
| `matplotlib` | Network graph rendering |
| `pdfplumber` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `python-pptx` | PPTX text extraction |
| `requests` | Anthropic Claude API calls |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

MIT License — see `LICENSE` for details.
