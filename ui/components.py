# =============================================================================
# ui/components.py  –  CSS injection, theme colours, and every Streamlit
#                       render helper used across the application.
# =============================================================================

import collections
import datetime
import string

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.data import COURSES, COURSE_SYLLABUS, DIFFICULTY_SETTINGS
from core.engine import (
    ai_recommendations,
    COGNITIVE_PATTERNS,
    compute_behavioral_features,
    compute_per_quiz_behavior,
    classify_cognitive_pattern,
    generate_adaptive_recommendations,
    generate_improvement_report,
)


# =============================================================================
# THEME COLOURS
# =============================================================================

def build_theme(dark_mode: bool) -> dict:
    DM = dark_mode
    return {
        "DM":         DM,
        "BG_MAIN":    "#1a1a2e" if DM else "#f5f4ef",
        "BG_CARD":    "#252538" if DM else "#ffffff",
        "BG_CARD2":   "#2a2a40" if DM else "#faf9f5",
        "BG_SIDE":    "#0f0f1a" if DM else "#1c1c28",
        "CLR_TEXT":   "#e0e0f0" if DM else "#1a1a2e",
        "CLR_SUB":    "#a0a0c0" if DM else "#6b6860",
        "CLR_BORDER": "#3a3a5a" if DM else "#e8e6df",
        "CLR_SOFT":   "#2e2e4a" if DM else "#f0ede6",
    }


# =============================================================================
# CSS INJECTION
# =============================================================================

def inject_css(t: dict) -> None:
    BG_MAIN    = t["BG_MAIN"]
    BG_CARD    = t["BG_CARD"]
    BG_CARD2   = t["BG_CARD2"]
    BG_SIDE    = t["BG_SIDE"]
    CLR_TEXT   = t["CLR_TEXT"]
    CLR_SUB    = t["CLR_SUB"]
    CLR_BORDER = t["CLR_BORDER"]
    CLR_SOFT   = t["CLR_SOFT"]

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Literata:ital,wght@0,400;0,600;0,700;1,400&display=swap');
*,*::before,*::after{{font-family:'Plus Jakarta Sans',sans-serif;box-sizing:border-box;}}
h1,h2,h3{{font-family:'Literata',serif;}}
.stApp{{background:{BG_MAIN};color:{CLR_TEXT};}}
[data-testid="stSidebar"]{{background:{BG_SIDE} !important;border-right:none !important;}}
[data-testid="stSidebar"] *{{color:#d0d0e0 !important;}}
[data-testid="stSidebar"] h3{{color:#ffffff !important;font-family:'Plus Jakarta Sans',sans-serif !important;}}
.stTabs [data-baseweb="tab-list"]{{background:{BG_CARD};padding:6px 8px;border-radius:12px;border:1px solid {CLR_BORDER};gap:4px;margin-bottom:24px;box-shadow:0 2px 8px rgba(0,0,0,0.06);}}
.stTabs [data-baseweb="tab"]{{font-size:13px;font-weight:600;color:{CLR_SUB};padding:9px 22px;border-radius:8px;background:transparent;border:none;transition:all 0.2s;}}
.stTabs [aria-selected="true"]{{color:{CLR_TEXT} !important;background:{BG_MAIN} !important;box-shadow:0 1px 4px rgba(0,0,0,0.12);border:1px solid {CLR_BORDER};}}
.card{{background:{BG_CARD};border-radius:16px;border:1px solid {CLR_BORDER};padding:24px;box-shadow:0 2px 8px rgba(0,0,0,0.04);transition:box-shadow .25s,transform .25s;}}
.card:hover{{box-shadow:0 8px 24px rgba(0,0,0,0.12);transform:translateY(-2px);}}
.metric-card{{background:{BG_CARD};border-radius:14px;border:1px solid {CLR_BORDER};padding:20px 22px;position:relative;overflow:hidden;box-shadow:0 2px 6px rgba(0,0,0,0.04);}}
.metric-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;}}
.metric-card.indigo::before{{background:#4f46e5;}}
.metric-card.green::before{{background:#16a34a;}}
.metric-card.amber::before{{background:#d97706;}}
.metric-card.rose::before{{background:#e11d48;}}
.metric-card.sky::before{{background:#0ea5e9;}}
.metric-card.violet::before{{background:#7c3aed;}}
.metric-card.teal::before{{background:#0d9488;}}
.grad{{background:linear-gradient(135deg,#4f46e5,#0ea5e9);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800;}}
.stButton>button{{font-family:'Plus Jakarta Sans',sans-serif !important;font-weight:600 !important;border-radius:10px !important;transition:all .2s !important;font-size:13px !important;}}
.stButton>button[kind="primary"]{{background:#4f46e5 !important;border:none !important;color:#fff !important;box-shadow:0 4px 12px rgba(79,70,229,.28) !important;}}
.stButton>button[kind="primary"]:hover{{background:#4338ca !important;transform:translateY(-1px) !important;}}
.stTextInput>div>div>input,.stNumberInput>div>div>input{{background:{BG_CARD2} !important;border:1px solid {CLR_BORDER} !important;color:{CLR_TEXT} !important;border-radius:10px !important;}}
.stSelectbox>div>div{{background:{BG_CARD2} !important;border:1px solid {CLR_BORDER} !important;border-radius:10px !important;}}
.stRadio label,.stCheckbox label{{color:{CLR_TEXT} !important;}}
[data-testid="stExpander"]{{background:{BG_CARD};border:1px solid {CLR_BORDER} !important;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,0.04);}}
[data-testid="stExpander"] summary{{color:{CLR_TEXT} !important;font-weight:600 !important;}}
[data-testid="stFileUploadDropzone"]{{background:{BG_CARD2} !important;border:2px dashed {CLR_BORDER} !important;border-radius:12px !important;}}
[data-testid="stFileUploadDropzone"]:hover{{border-color:#4f46e5 !important;}}
.stProgress>div>div>div{{background:#4f46e5 !important;}}
.stAlert{{border-radius:12px !important;}}
p,div,span,label{{color:{CLR_TEXT};}}
::-webkit-scrollbar{{width:7px;}}
::-webkit-scrollbar-track{{background:{CLR_SOFT};}}
::-webkit-scrollbar-thumb{{background:#c8c4bc;border-radius:4px;}}
::-webkit-scrollbar-thumb:hover{{background:#4f46e5;}}
.qcard{{background:{BG_CARD};border:1.5px solid {CLR_BORDER};border-radius:14px;padding:22px 26px;margin-bottom:14px;box-shadow:0 2px 6px rgba(0,0,0,0.04);}}
.qcard:hover{{border-color:#a5b4fc;box-shadow:0 4px 14px rgba(79,70,229,.08);}}
.badge{{display:inline-block;padding:3px 11px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:.5px;margin-bottom:12px;}}
.badge-mcq{{background:#ede9fe;color:#5b21b6;border:1px solid #c4b5fd;}}
.badge-msq{{background:#dcfce7;color:#15803d;border:1px solid #86efac;}}
.badge-blank{{background:#fff7ed;color:#c2410c;border:1px solid #fed7aa;}}
.badge-easy{{background:#dcfce7;color:#15803d;border:1px solid #86efac;}}
.badge-medium{{background:#fef9c3;color:#a16207;border:1px solid #fde047;}}
.badge-hard{{background:#fee2e2;color:#991b1b;border:1px solid #fca5a5;}}
.exam-docx{{background:#ffffff;color:#111;padding:72px 88px;border-radius:4px;border:1px solid #ccc;box-shadow:0 4px 24px rgba(0,0,0,.13);font-family:'Times New Roman',Georgia,serif;max-width:920px;margin:0 auto;line-height:1.9;}}
.exam-docx .exam-inst{{background:#f9f7f0;border:1px solid #ddd;padding:14px 20px;border-radius:6px;font-size:13px;margin-bottom:32px;font-style:italic;}}
.exam-docx .part-header{{font-size:15px;font-weight:700;text-align:center;text-decoration:underline;letter-spacing:1px;margin:36px 0 6px;text-transform:uppercase;}}
.exam-docx .part-sub{{font-size:12.5px;font-style:italic;text-align:center;color:#555;margin-bottom:20px;}}
.exam-docx .exam-q-row{{display:grid;grid-template-columns:28px 1fr 52px;gap:0 10px;margin-bottom:22px;padding-bottom:18px;border-bottom:1px dashed #ddd;font-size:14px;align-items:start;}}
.exam-docx .exam-q-num{{font-weight:700;padding-top:2px;}}
.exam-docx .exam-q-text{{line-height:1.85;}}
.exam-docx .exam-q-mark{{font-weight:700;text-align:right;color:#444;white-space:nowrap;padding-top:2px;}}
.exam-docx .exam-q-sub{{font-size:12.5px;color:#555;margin-top:5px;font-style:italic;grid-column:2/3;}}
.lesson-row{{display:flex;align-items:center;gap:12px;padding:10px 16px;border-radius:10px;margin-bottom:6px;border:1px solid {CLR_BORDER};background:{BG_CARD};transition:all .2s;}}
.lesson-row:hover{{border-color:#a5b4fc;}}
.lesson-done{{background:#f0fdf4;border-color:#86efac;}}
.lesson-active{{background:#eff6ff;border-color:#93c5fd;}}
.rec-card{{background:{BG_CARD};border:1px solid {CLR_BORDER};border-radius:14px;padding:20px 22px;margin-bottom:14px;position:relative;overflow:hidden;box-shadow:0 2px 6px rgba(0,0,0,0.04);}}
.rec-card::before{{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;border-radius:14px 0 0 14px;}}
.rec-high::before{{background:#ef4444;}}
.rec-med::before{{background:#f59e0b;}}
.rec-low::before{{background:#22c55e;}}
.note-box{{background:{BG_CARD2};border:1px solid {CLR_BORDER};border-left:4px solid #4f46e5;border-radius:10px;padding:14px 18px;font-size:13.5px;line-height:1.75;color:{CLR_TEXT};margin-bottom:12px;}}
.streak-box{{display:inline-flex;align-items:center;gap:6px;background:#fff7ed;border:1px solid #fed7aa;border-radius:20px;padding:5px 14px;font-size:13px;font-weight:700;color:#c2410c;}}
.summary-tag{{display:inline-block;background:#ede9fe;color:#5b21b6;border:1px solid #c4b5fd;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;margin:3px 4px 3px 0;}}
.pattern-card{{border-radius:18px;padding:24px 28px;margin-bottom:20px;position:relative;overflow:hidden;}}
.evidence-tag{{display:inline-block;background:rgba(79,70,229,0.08);border:1px solid rgba(79,70,229,0.2);color:#4f46e5;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;margin-top:6px;}}
.milestone-badge{{display:inline-flex;align-items:center;gap:6px;background:{BG_CARD2};border:1px solid {CLR_BORDER};border-radius:12px;padding:8px 14px;font-size:13px;font-weight:600;margin:4px;}}
@keyframes pulse-red{{0%,100%{{color:#ef4444;transform:scale(1);}}50%{{color:#dc2626;transform:scale(1.06);}}}}
@keyframes tick{{0%{{opacity:1;}}50%{{opacity:.6;}}100%{{opacity:1;}}}}
.timer-display{{font-size:2.8rem;font-weight:900;font-family:'Literata',serif;color:#4f46e5;letter-spacing:2px;transition:color .3s;animation:tick 1s infinite;}}
.timer-display.warning{{color:#d97706;animation:tick .6s infinite;}}
.timer-display.danger{{color:#ef4444;animation:pulse-red .5s infinite;}}
.stRadio [role="radiogroup"] label{{border:1px solid {CLR_BORDER};border-radius:10px;padding:9px 16px;margin-bottom:6px;display:block;cursor:pointer;transition:all .18s;background:{BG_CARD};}}
.stRadio [role="radiogroup"] label:hover{{border-color:#a5b4fc;background:#f5f3ff;}}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SUMMARY RENDERER
# =============================================================================

def render_summary(sdata: dict, t: dict, color: str = "#4f46e5") -> None:
    CLR_TEXT   = t["CLR_TEXT"]
    CLR_SUB    = t["CLR_SUB"]
    CLR_BORDER = t["CLR_BORDER"]
    BG_CARD2   = t["BG_CARD2"]

    if not sdata:
        st.info("No summary data available.")
        return

    ov = sdata.get("document_overview", "")
    if ov:
        st.markdown(f'<div class="note-box" style="border-left-color:{color};">{ov}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<h4 style="color:{color};font-size:14px;font-weight:700;margin-bottom:10px;">📌 Key Topics</h4>', unsafe_allow_html=True)
        for topic in sdata.get("key_topics", []):
            st.markdown(
                f'<div style="background:{BG_CARD2};border:1px solid {CLR_BORDER};border-radius:8px;'
                f'padding:8px 14px;margin-bottom:6px;font-size:13.5px;">• {topic}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(f'<h4 style="color:{color};font-size:14px;font-weight:700;margin:16px 0 10px;">⭐ Exam Topics</h4>', unsafe_allow_html=True)
        tags = "".join(f'<span class="summary-tag">{x}</span>' for x in sdata.get("exam_topics", []))
        st.markdown(tags, unsafe_allow_html=True)

    with col2:
        st.markdown(f'<h4 style="color:{color};font-size:14px;font-weight:700;margin-bottom:10px;">📖 Core Definitions</h4>', unsafe_allow_html=True)
        for d in sdata.get("core_definitions", []):
            st.markdown(
                f'<div style="display:flex;gap:8px;margin-bottom:8px;font-size:13.5px;">'
                f'<span style="color:{color};font-weight:700;min-width:130px;">{d.get("term","")}</span>'
                f'<span style="flex:1;">{d.get("definition","")}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown(f'<h4 style="color:{color};font-size:14px;font-weight:700;margin:16px 0 10px;">⚡ Quick Revision</h4>', unsafe_allow_html=True)
        for pt in sdata.get("quick_review", []):
            st.markdown(
                f'<div style="font-size:13px;padding:4px 0;border-bottom:1px dashed {CLR_BORDER};">→ {pt}</div>',
                unsafe_allow_html=True,
            )

    principles = sdata.get("important_principles", [])
    if principles:
        st.markdown(f'<h4 style="color:{color};font-size:14px;font-weight:700;margin:16px 0 10px;">⚖️ Key Principles</h4>', unsafe_allow_html=True)
        for p in principles:
            st.markdown(f'<div class="note-box">— {p}</div>', unsafe_allow_html=True)


# =============================================================================
# COGNITIVE PATTERN CARD RENDERER
# =============================================================================

def render_pattern_card(pattern: str, features: dict, t: dict) -> None:
    """Display the classified cognitive pattern with behavioral evidence."""
    BG_CARD    = t["BG_CARD"]
    BG_CARD2   = t["BG_CARD2"]
    CLR_TEXT   = t["CLR_TEXT"]
    CLR_SUB    = t["CLR_SUB"]
    CLR_BORDER = t["CLR_BORDER"]

    info  = COGNITIVE_PATTERNS.get(pattern, {})
    color = info.get("color", "#4f46e5")
    icon  = info.get("icon",  "🧠")
    desc  = info.get("desc",  "")

    acc       = (1 - features.get("mistake_frequency", 0.5)) * 100
    avg_time  = features.get("avg_response_time_s", 0)
    retry     = features.get("avg_retry_rate", 0)
    slope     = features.get("improvement_slope", 0)
    cons      = features.get("topic_consistency", 0) * 100
    avg_score = features.get("avg_score", 0)

    st.markdown(f"""
    <div style="background:{BG_CARD};border:2px solid {color};border-radius:18px;padding:24px 28px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:12px;">
            <div style="width:60px;height:60px;background:{color}22;border:2px solid {color};
                        border-radius:16px;display:flex;align-items:center;justify-content:center;
                        font-size:28px;">{icon}</div>
            <div>
                <div style="font-size:11px;font-weight:700;letter-spacing:2px;color:{color};text-transform:uppercase;">Cognitive Learning Pattern</div>
                <div style="font-size:22px;font-weight:800;color:{CLR_TEXT};font-family:'Literata',serif;">{pattern}</div>
            </div>
        </div>
        <div style="color:{CLR_SUB};font-size:13.5px;line-height:1.6;margin-bottom:16px;border-left:3px solid {color};padding-left:12px;">{desc}</div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;">
            <div style="background:{BG_CARD2};border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:1.4rem;font-weight:800;color:{color};">{acc:.0f}%</div>
                <div style="font-size:11px;color:{CLR_SUB};">Accuracy</div>
            </div>
            <div style="background:{BG_CARD2};border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:1.4rem;font-weight:800;color:{color};">{avg_time:.0f}s</div>
                <div style="font-size:11px;color:{CLR_SUB};">Avg Response Time</div>
            </div>
            <div style="background:{BG_CARD2};border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:1.4rem;font-weight:800;color:{color};">{retry:.1f}×</div>
                <div style="font-size:11px;color:{CLR_SUB};">Avg Retries</div>
            </div>
            <div style="background:{BG_CARD2};border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:1.4rem;font-weight:800;color:{'#22c55e' if slope>=0 else '#ef4444'};">{'+' if slope>=0 else ''}{slope:.1f}%</div>
                <div style="font-size:11px;color:{CLR_SUB};">Improvement/Quiz</div>
            </div>
            <div style="background:{BG_CARD2};border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:1.4rem;font-weight:800;color:{color};">{cons:.0f}%</div>
                <div style="font-size:11px;color:{CLR_SUB};">Topic Consistency</div>
            </div>
            <div style="background:{BG_CARD2};border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:1.4rem;font-weight:800;color:{color};">{avg_score:.0f}%</div>
                <div style="font-size:11px;color:{CLR_SUB};">Avg Score</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# BEHAVIORAL ANALYTICS RENDERER  (full section for Analytics tab)
# =============================================================================

def render_behavioral_analytics(quiz_history: list, retry_map: dict, overall_progress: int, t: dict) -> None:
    """
    Render the full Behavioral Analysis section:
    - Behavioral features table
    - Cognitive pattern classification card
    - Response-time trend
    - Mistake frequency trend
    - Retry pattern chart
    - Topic consistency radar
    """
    BG_CARD    = t["BG_CARD"]
    BG_CARD2   = t["BG_CARD2"]
    CLR_TEXT   = t["CLR_TEXT"]
    CLR_SUB    = t["CLR_SUB"]
    CLR_BORDER = t["CLR_BORDER"]
    CLR_SOFT   = t["CLR_SOFT"]

    if not quiz_history:
        st.info("🧠 Complete at least one quiz to unlock Behavioral Analysis.")
        return

    features = compute_behavioral_features(quiz_history, retry_map)
    pattern  = classify_cognitive_pattern(features)

    st.markdown("### 🧠 Cognitive Pattern Classification")
    render_pattern_card(pattern, features, t)

    # ── Behavioral features table
    st.markdown("#### 📊 Extracted Behavioral Features")
    feat_labels = {
        "avg_response_time_s":  ("Avg Response Time (s)",  "Time per question — indicates confidence & processing speed"),
        "mistake_frequency":    ("Mistake Frequency",       "Proportion of incorrect answers — lower is better"),
        "avg_retry_rate":       ("Avg Retry Rate",          "How many times you repeat quizzes — reflects effort"),
        "speed_accuracy_ratio": ("Speed-Accuracy Ratio",    "Combined index of speed and correctness"),
        "topic_consistency":    ("Topic Consistency",       "Uniformity of performance across topics — higher is better"),
        "improvement_slope":    ("Improvement Slope (%/q)", "Rate of score change per attempt — positive = improving"),
        "hard_difficulty_bias": ("Hard Difficulty Bias",    "Fraction of attempts on Hard quizzes"),
        "avg_score":            ("Average Score (%)",       "Mean quiz score across all attempts"),
    }
    feat_rows = []
    for k, (label, hint) in feat_labels.items():
        val = features.get(k, 0)
        feat_rows.append({"Feature": label, "Value": val, "What it means": hint})

    import pandas as pd
    df_feat = pd.DataFrame(feat_rows)
    st.dataframe(
        df_feat.set_index("Feature"),
        use_container_width=True,
        height=300,
    )

    st.markdown("---")

    # ── Per-quiz behavioral trends
    per_q_data = [compute_per_quiz_behavior(q) for q in quiz_history]
    attempts   = list(range(1, len(per_q_data) + 1))

    if len(per_q_data) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ⏱ Response Time per Attempt")
            times = [p["time_per_q_s"] for p in per_q_data]
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=attempts, y=times, mode="lines+markers+text",
                line=dict(color="#0ea5e9", width=2.5),
                marker=dict(size=9, color="#0ea5e9", line=dict(color="#fff", width=2)),
                text=[f"{v:.0f}s" for v in times], textposition="top center",
                textfont=dict(color=CLR_TEXT, size=11),
                fill="tozeroy", fillcolor="rgba(14,165,233,0.07)",
                name="Time/Q",
            ))
            fig_t.add_hline(y=60, line_dash="dash", line_color="#f59e0b",
                            annotation_text="60s target", annotation_font_color="#d97706")
            fig_t.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=260, margin=dict(l=0, r=0, t=10, b=10),
                xaxis=dict(title="Attempt #", tickfont=dict(color=CLR_SUB), gridcolor=CLR_SOFT),
                yaxis=dict(title="Seconds", tickfont=dict(color=CLR_SUB), gridcolor=CLR_SOFT),
                font=dict(color=CLR_TEXT), showlegend=False,
            )
            st.plotly_chart(fig_t, use_container_width=True)

        with col2:
            st.markdown("#### ❌ Mistake Frequency per Attempt")
            wrongs = [p["wrong_count"] for p in per_q_data]
            totals = [q.get("total", 1) for q in quiz_history]
            miss_pct = [w / max(t, 1) * 100 for w, t in zip(wrongs, totals)]
            fig_m = go.Figure()
            fig_m.add_trace(go.Bar(
                x=attempts, y=miss_pct,
                marker_color=["#ef4444" if v > 40 else "#f59e0b" if v > 20 else "#22c55e" for v in miss_pct],
                text=[f"{v:.0f}%" for v in miss_pct], textposition="outside",
                textfont=dict(color=CLR_TEXT, size=11),
                name="Mistake %",
            ))
            fig_m.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=260, margin=dict(l=0, r=0, t=10, b=10),
                xaxis=dict(title="Attempt #", tickfont=dict(color=CLR_SUB), gridcolor=CLR_SOFT),
                yaxis=dict(title="Mistake %", tickfont=dict(color=CLR_SUB), gridcolor=CLR_SOFT, range=[0, 110]),
                font=dict(color=CLR_TEXT), showlegend=False,
            )
            st.plotly_chart(fig_m, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### 🔁 Retry Patterns by Subject")
            if retry_map:
                subjects = list(retry_map.keys())
                retries  = list(retry_map.values())
                fig_r = go.Figure(go.Bar(
                    x=retries, y=[s[:22] for s in subjects],
                    orientation="h",
                    marker_color=["#8b5cf6" if r >= 3 else "#0ea5e9" if r >= 2 else "#22c55e" for r in retries],
                    text=retries, textposition="outside",
                    textfont=dict(color=CLR_TEXT),
                ))
                fig_r.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=max(200, len(subjects) * 44), margin=dict(l=10, r=50, t=10, b=10),
                    xaxis=dict(title="Times Attempted", tickfont=dict(color=CLR_SUB), gridcolor=CLR_SOFT),
                    yaxis=dict(tickfont=dict(color=CLR_TEXT, size=11)),
                    font=dict(color=CLR_TEXT),
                )
                st.plotly_chart(fig_r, use_container_width=True)
            else:
                st.info("Retry data will appear after attempting quizzes multiple times.")

        with col4:
            st.markdown("#### 🎯 Behavioral Profile Radar")
            # Normalize all features to 0-100 scale for radar
            acc_pct   = (1 - features.get("mistake_frequency", 0.5)) * 100
            speed_pct = min(100, (1 / (1 + features.get("avg_response_time_s", 60) / 60)) * 100)
            cons_pct  = features.get("topic_consistency", 0.5) * 100
            retry_inv = max(0, 100 - (features.get("avg_retry_rate", 1) - 1) * 20)
            improv    = min(100, max(0, 50 + features.get("improvement_slope", 0) * 5))
            cats  = ["Accuracy", "Speed", "Consistency", "First-Try\nSuccess", "Improvement"]
            vals  = [acc_pct, speed_pct, cons_pct, retry_inv, improv]
            info  = COGNITIVE_PATTERNS.get(pattern, {})
            pc    = info.get("color", "#4f46e5")
            fig_radar = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", fillcolor=f"{pc}22",
                line=dict(color=pc, width=2.5),
                marker=dict(color=pc, size=8),
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100],
                                    tickfont=dict(color=CLR_SUB, size=9), gridcolor=CLR_BORDER),
                    angularaxis=dict(tickfont=dict(color=CLR_TEXT, size=10), gridcolor=CLR_BORDER),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)", height=300,
                margin=dict(t=20, b=20, l=30, r=30),
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# =============================================================================
# ADAPTIVE RECOMMENDATION RENDERER
# =============================================================================

def render_adaptive_recommendations(
    quiz_history: list,
    retry_map: dict,
    overall_progress: int,
    api_key: str,
    t: dict,
) -> None:
    """Render locally-generated adaptive recommendations (no API required),
    with optional Claude AI enhancement."""
    BG_CARD2   = t["BG_CARD2"]
    CLR_TEXT   = t["CLR_TEXT"]
    CLR_SUB    = t["CLR_SUB"]
    CLR_BORDER = t["CLR_BORDER"]

    features = compute_behavioral_features(quiz_history, retry_map)
    pattern  = classify_cognitive_pattern(features)

    st.markdown("### 🎯 Adaptive Recommendation Engine")
    st.markdown(
        f'<div class="note-box" style="border-left-color:#4f46e5;">Based on your <strong>{pattern}</strong> '
        f'learning profile, here are data-driven strategies tailored to your behavioral patterns.</div>',
        unsafe_allow_html=True,
    )

    # Always generate local recommendations
    local_recs = generate_adaptive_recommendations(features, pattern, quiz_history, retry_map, overall_progress)

    # Optionally enhance with Claude
    if api_key and quiz_history:
        last = quiz_history[-1]
        with st.spinner("Enhancing with Claude AI personalisation…"):
            ai_recs = ai_recommendations(
                last.get("responses", {}), overall_progress,
                last.get("subject", ""), api_key
            )
        if ai_recs:
            # Merge: local provides behavioral evidence, AI provides subject-specific guidance
            for rec in ai_recs[:2]:
                rec["evidence"] = "Claude AI — subject-specific analysis"
                local_recs.insert(0, rec)
            local_recs = local_recs[:6]

    cols = st.columns(2)
    for i, rec in enumerate(local_recs):
        pri = rec.get("priority", "medium")
        pc  = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}.get(pri, "#4f46e5")
        pl  = {"high": "🔴 HIGH", "medium": "🟡 MED", "low": "🟢 LOW"}.get(pri, "⚪")
        ev  = rec.get("evidence", "")
        with cols[i % 2]:
            st.markdown(f"""
            <div class="rec-card rec-{pri[:3]}">
                <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                    <div style="font-weight:700;font-size:14px;">{rec.get('title','')}</div>
                    <span style="font-size:11px;font-weight:700;color:{pc};">{pl}</span>
                </div>
                <div style="font-size:13px;line-height:1.6;margin-bottom:8px;">{rec.get('description','')}</div>
                <div style="background:{BG_CARD2};border-radius:8px;padding:8px 12px;">
                    <div style="font-size:12px;color:#4f46e5;margin-bottom:3px;">⚡ {rec.get('action','')}</div>
                    <div style="font-size:12px;color:{pc};font-weight:600;">⏱ {rec.get('time_est','')} · 📈 {rec.get('impact','')}</div>
                    {f'<div class="evidence-tag">📊 {ev}</div>' if ev else ''}
                </div>
            </div>""", unsafe_allow_html=True)


# =============================================================================
# IMPROVEMENT ANALYTICS REPORT RENDERER
# =============================================================================

def render_improvement_report(
    quiz_history: list,
    retry_map: dict,
    overall_progress: int,
    t: dict,
) -> None:
    """Render the full Improvement Analytics Report."""
    BG_CARD    = t["BG_CARD"]
    BG_CARD2   = t["BG_CARD2"]
    CLR_TEXT   = t["CLR_TEXT"]
    CLR_SUB    = t["CLR_SUB"]
    CLR_BORDER = t["CLR_BORDER"]
    CLR_SOFT   = t["CLR_SOFT"]

    if not quiz_history:
        st.info("Complete at least one quiz to generate your Improvement Analytics Report.")
        return

    features = compute_behavioral_features(quiz_history, retry_map)
    pattern  = classify_cognitive_pattern(features)
    report   = generate_improvement_report(quiz_history, retry_map, features, pattern, overall_progress)

    if "error" in report:
        st.error(report["error"])
        return

    # ── Summary KPIs
    st.markdown("### 📋 Improvement Analytics Report")
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_data = [
        (k1, "Avg Score",         f"{report['avg_score']}%",      "green"),
        (k2, "Best Score",        f"{report['best_score']}%",     "indigo"),
        (k3, "WoW Change",        f"{'+' if report['wow_change']>=0 else ''}{report['wow_change']}%", "amber"),
        (k4, "Predicted Next",    f"{report['predicted_next']}%", "sky"),
        (k5, "Total Quizzes",     str(report["total_quizzes"]),   "violet"),
    ]
    for col, title, val, cls in kpi_data:
        with col:
            st.markdown(f"""
            <div class="metric-card {cls}" style="text-align:center;">
                <div style="color:{CLR_SUB};font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;">{title}</div>
                <div style="font-size:1.8rem;font-weight:800;color:{CLR_TEXT};font-family:'Literata',serif;margin-top:6px;">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Score trajectory
    st.markdown("#### 📈 Score Trajectory & Prediction")
    import pandas as pd
    all_scores = [q.get("score", 0) for q in quiz_history]
    attempts   = list(range(1, len(all_scores) + 1))
    slope      = features.get("improvement_slope", 0)
    pred_score = min(100, max(0, all_scores[-1] + slope)) if all_scores else 0

    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(
        x=attempts, y=all_scores, mode="lines+markers",
        line=dict(color="#4f46e5", width=2.5),
        marker=dict(size=9, color="#4f46e5", line=dict(color="#fff", width=2)),
        fill="tozeroy", fillcolor="rgba(79,70,229,0.07)",
        name="Actual Score",
    ))
    if len(all_scores) >= 2:
        # Trend line
        x_ext = attempts + [len(all_scores) + 1]
        trend  = np.polyval(np.polyfit(attempts, all_scores, 1), x_ext)
        fig_traj.add_trace(go.Scatter(
            x=x_ext, y=trend.tolist(), mode="lines",
            line=dict(color="#f59e0b", width=1.5, dash="dash"),
            name="Trend",
        ))
        # Prediction marker
        fig_traj.add_trace(go.Scatter(
            x=[len(all_scores) + 1], y=[pred_score],
            mode="markers+text",
            marker=dict(size=14, color="#22c55e", symbol="star",
                        line=dict(color="#fff", width=2)),
            text=[f"Predicted: {pred_score:.0f}%"],
            textposition="top center",
            textfont=dict(color="#22c55e", size=12, family="Plus Jakarta Sans"),
            name="Prediction",
        ))
    fig_traj.add_hline(y=70, line_dash="dash", line_color="#ef4444",
                       annotation_text="Pass threshold", annotation_font_color="#dc2626")
    fig_traj.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(title="Attempt #", tickfont=dict(color=CLR_SUB), gridcolor=CLR_SOFT),
        yaxis=dict(title="Score %", range=[0, 110], tickfont=dict(color=CLR_SUB), gridcolor=CLR_SOFT),
        legend=dict(font=dict(color=CLR_SUB)), font=dict(color=CLR_TEXT),
    )
    st.plotly_chart(fig_traj, use_container_width=True)

    # ── Milestones
    milestones = report.get("milestones", [])
    if milestones:
        st.markdown("#### 🏆 Achievements Unlocked")
        ms_html = "".join(
            f'<span class="milestone-badge">{m["icon"]} <strong>{m["label"]}</strong> — {m["desc"]}</span>'
            for m in milestones
        )
        st.markdown(ms_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Strengths & Weaknesses
    col_s, col_w = st.columns(2)
    with col_s:
        st.markdown("#### ✅ Behavioral Strengths")
        for s in report.get("strengths", ["No strengths identified yet — complete more quizzes."]):
            st.markdown(
                f'<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:10px;'
                f'padding:10px 14px;margin-bottom:8px;font-size:13.5px;">✅ {s}</div>',
                unsafe_allow_html=True,
            )
    with col_w:
        st.markdown("#### ⚠️ Improvement Areas")
        for w in report.get("weaknesses", ["No weaknesses detected yet."]):
            st.markdown(
                f'<div style="background:#fff1f2;border:1px solid #fca5a5;border-radius:10px;'
                f'padding:10px 14px;margin-bottom:8px;font-size:13.5px;">⚠️ {w}</div>',
                unsafe_allow_html=True,
            )

    # ── Topic analysis
    col_st, col_wt = st.columns(2)
    with col_st:
        if report.get("strong_topics"):
            st.markdown("#### 🟢 Strongest Topics")
            for tp in report["strong_topics"]:
                st.markdown(
                    f'<span class="summary-tag">💪 {tp}</span>',
                    unsafe_allow_html=True,
                )
    with col_wt:
        if report.get("weak_topics"):
            st.markdown("#### 🔴 Weakest Topics")
            for tp in report["weak_topics"]:
                st.markdown(
                    f'<div style="display:inline-block;background:#fee2e2;color:#991b1b;'
                    f'border:1px solid #fca5a5;padding:4px 12px;border-radius:20px;'
                    f'font-size:12px;font-weight:600;margin:3px 4px 3px 0;">⚠️ {tp}</div>',
                    unsafe_allow_html=True,
                )


# =============================================================================
# QUIZ RENDERER
# =============================================================================

def render_quiz(quiz: list, ex_id: str, time_lim: int, subject: str,
                xp: int, difficulty: str, t: dict) -> None:
    ts  = st.session_state["training_state"]
    cfg = DIFFICULTY_SETTINGS[difficulty]

    BG_CARD    = t["BG_CARD"]
    BG_CARD2   = t["BG_CARD2"]
    CLR_TEXT   = t["CLR_TEXT"]
    CLR_SUB    = t["CLR_SUB"]
    CLR_BORDER = t["CLR_BORDER"]
    CLR_SOFT   = t["CLR_SOFT"]

    starred   = subject in st.session_state.get("starred_quizzes", [])
    star_icon = "⭐" if starred else "☆"
    star_lbl  = "Starred!" if starred else "Star this Quiz"
    hdr1, hdr2 = st.columns([5, 1])
    with hdr2:
        if st.button(f"{star_icon} {star_lbl}", key="star_quiz_btn", use_container_width=True):
            sq = st.session_state.setdefault("starred_quizzes", [])
            if subject in sq:
                sq.remove(subject)
                st.toast("Removed from starred quizzes")
            else:
                sq.append(subject)
                st.toast("⭐ Quiz starred!")
            st.rerun()

    if not ts.get("quiz_start_ts"):
        ts["quiz_start_ts"] = datetime.datetime.now().isoformat()

    total_secs = time_lim * 60
    start_iso  = ts["quiz_start_ts"]
    hdr1.markdown(f"""
    <div style="background:{BG_CARD};border:1.5px solid {CLR_BORDER};border-radius:14px;padding:20px 28px;
                margin-bottom:24px;display:flex;justify-content:space-between;align-items:center;
                box-shadow:0 2px 8px rgba(0,0,0,.05);">
        <div>
            <div style="font-size:20px;font-weight:800;color:{CLR_TEXT};font-family:'Literata',serif;">⏱ Active Quiz</div>
            <div style="color:{CLR_SUB};font-size:13px;margin-top:4px;">{len(quiz)} Questions · {time_lim} mins · {subject}</div>
            <span class="badge badge-{difficulty.lower()}" style="margin-top:8px;">{difficulty} · {cfg['bonus_xp']}× XP</span>
        </div>
        <div style="text-align:right;">
            <div id="countdown-timer" class="timer-display">--:--</div>
            <div style="color:{CLR_SUB};font-size:11px;">Time Remaining</div>
            <div id="timer-bar-wrap" style="width:140px;height:5px;background:{CLR_SOFT};border-radius:3px;margin-top:6px;overflow:hidden;">
                <div id="timer-bar" style="height:100%;background:#4f46e5;border-radius:3px;width:100%;transition:width .9s linear;"></div>
            </div>
        </div>
    </div>
    <script>
    (function() {{
        const startIso = "{start_iso}";
        const totalMs  = {total_secs} * 1000;
        const startTs  = new Date(startIso).getTime();
        function fmt(s) {{
            let m = Math.floor(s/60), sec = s % 60;
            return String(m).padStart(2,'0') + ':' + String(sec).padStart(2,'0');
        }}
        function tick() {{
            const elapsed = Date.now() - startTs;
            const rem     = Math.max(0, totalMs - elapsed);
            const remSecs = Math.ceil(rem / 1000);
            const pct     = (rem / totalMs) * 100;
            const el      = document.getElementById('countdown-timer');
            const bar     = document.getElementById('timer-bar');
            if (!el) return;
            el.textContent = fmt(remSecs);
            el.className   = 'timer-display' + (remSecs <= 60 ? ' danger' : remSecs <= 180 ? ' warning' : '');
            if (bar) bar.style.width = pct + '%';
            if (bar) bar.style.background = remSecs <= 60 ? '#ef4444' : remSecs <= 180 ? '#d97706' : '#4f46e5';
            if (rem > 0) setTimeout(tick, 800);
        }}
        tick();
    }})();
    </script>
    """, unsafe_allow_html=True)

    responses = {}
    for q in quiz:
        qtype      = q.get("type", "MCQ")
        diff       = q.get("difficulty", difficulty)
        topic      = q.get("topic", "General")
        badge_type = {"MCQ": "badge-mcq", "MSQ": "badge-msq", "BLANK": "badge-blank"}.get(qtype, "badge-mcq")
        label_type = {"MCQ": "Multiple Choice", "MSQ": "Multi-Select", "BLANK": "Fill in Blank"}.get(qtype, qtype)

        st.markdown(f"""
        <div class="qcard">
          <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:10px;">
            <span class="badge {badge_type}">{label_type}</span>
            <span class="badge badge-{diff.lower()}">{diff}</span>
            <span style="background:{BG_CARD2};border:1px solid {CLR_BORDER};border-radius:20px;
                         padding:3px 11px;font-size:11px;font-weight:600;">📍 {topic[:35]}</span>
          </div>
          <div style="font-size:15.5px;font-weight:600;line-height:1.7;color:{CLR_TEXT};">
            <span style="color:#4f46e5;font-weight:800;font-size:16px;">Q{q['num']}.</span>&nbsp;{q['question']}
          </div>
        </div>""", unsafe_allow_html=True)

        opts        = q.get("options", ["Option A", "Option B", "Option C", "Option D"])
        correct_ans = q.get("answer", "A")

        if qtype == "MCQ":
            choice   = st.radio("", opts, key=f"q_{q['num']}", index=None, label_visibility="collapsed")
            is_ok    = False
            user_ans = ""
            if choice is not None:
                idx      = opts.index(choice)
                user_ans = chr(65 + idx)
                is_ok    = user_ans == correct_ans
            responses[q["num"]] = {
                "answer": choice or "", "correct": is_ok,
                "topic": topic, "explanation": q.get("explanation", ""),
                "difficulty": diff,
            }
        elif qtype == "MSQ":
            st.markdown(f'<div style="font-size:12px;color:{CLR_SUB};margin-bottom:6px;">Select all that apply</div>', unsafe_allow_html=True)
            sel_letters = []
            cols_q = st.columns(2)
            for oi, opt in enumerate(opts):
                with cols_q[oi % 2]:
                    if st.checkbox(opt, key=f"qo_{q['num']}_{oi}", value=False):
                        sel_letters.append(chr(65 + oi))
            correct_set = set(correct_ans) if isinstance(correct_ans, list) else {correct_ans}
            is_ok = set(sel_letters) == correct_set
            responses[q["num"]] = {
                "answer": sel_letters or ["(none)"], "correct": is_ok,
                "topic": topic, "explanation": q.get("explanation", ""),
                "difficulty": diff,
            }
        else:  # BLANK
            ans   = st.text_input("✏️ Fill in the blank:", key=f"q_{q['num']}", placeholder="Type your answer here…")
            is_ok = ans.strip().lower() == str(correct_ans).strip().lower()
            responses[q["num"]] = {
                "answer": ans, "correct": is_ok, "topic": topic,
                "explanation": q.get("explanation", ""), "difficulty": diff,
            }

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    earned_xp = int(xp * cfg["bonus_xp"])
    answered  = sum(1 for r in responses.values() if r.get("answer") not in ("", [], ["(none)"]))
    sc1, sc2  = st.columns([3, 2])
    with sc1:
        st.markdown(
            f'<div style="font-size:13px;color:{CLR_SUB};padding:10px 0;">📝 {answered}/{len(quiz)} answered</div>',
            unsafe_allow_html=True,
        )
    with sc2:
        if st.button(f"✅ Submit Quiz · Claim {earned_xp} XP", type="primary", use_container_width=True):
            ts["quiz_responses"] = responses
            ts["submitted"]      = True
            correct_n = sum(1 for r in responses.values() if r.get("correct"))

            dur_secs = 0
            if ts.get("quiz_start_ts"):
                dur_secs = (
                    datetime.datetime.now()
                    - datetime.datetime.fromisoformat(ts["quiz_start_ts"])
                ).total_seconds()

            st.session_state["global_xp"] += earned_xp
            qh_entry = {
                "subject":    subject,
                "xp":         earned_xp,
                "difficulty": difficulty,
                "score":      correct_n / len(responses) * 100 if responses else 0,
                "total":      len(responses),
                "correct":    correct_n,
                "wrong":      len(responses) - correct_n,
                "responses":  responses,
                "duration_s": int(dur_secs),
                "ts":         datetime.datetime.now().isoformat(timespec="seconds"),
            }
            st.session_state["quiz_history"].append(qh_entry)

            rc = st.session_state.setdefault("quiz_retry_count", {})
            rc[subject] = rc.get(subject, 0) + 1

            st.rerun()


# =============================================================================
# EXAM PAPER RENDERER
# =============================================================================

def render_exam_paper(edata: dict, subject: str, t: dict) -> None:
    pa    = edata.get("part_a", [])
    pb    = edata.get("part_b", [])
    total = sum(q["marks"] for q in pa + pb)
    subj  = edata.get("subject", subject).upper()
    year  = datetime.date.today().year
    inst  = edata.get("institution", "Innoverse College of Engineering & Technology")

    st.info("💡 Ctrl+P → Save as PDF")
    st.markdown(f"""
    <div class="exam-docx">
      <div style="text-align:center;border-bottom:3px double #111;padding-bottom:22px;margin-bottom:26px;">
        <div style="font-size:13px;letter-spacing:1.5px;font-weight:400;margin-bottom:4px;">AUTONOMOUS INSTITUTION – AFFILIATED TO ANNA UNIVERSITY</div>
        <div style="font-size:22px;font-weight:700;letter-spacing:1px;margin:6px 0 2px;font-family:'Times New Roman',serif;">{inst.upper()}</div>
        <div style="font-size:13px;color:#555;margin-bottom:10px;">Department of Computer Science and Engineering</div>
        <div style="font-size:17px;font-weight:700;letter-spacing:2px;text-transform:uppercase;border:2px solid #111;display:inline-block;padding:4px 24px;margin-top:6px;">MODEL EXAMINATION — {year}</div>
      </div>
      <table style="width:100%;font-size:13.5px;margin-bottom:24px;border-collapse:collapse;">
        <tr>
          <td style="width:60%;"><b>Subject Code &amp; Title:</b>&nbsp;{subj}</td>
          <td style="text-align:right;"><b>Date:</b>&nbsp;{datetime.date.today().strftime('%d / %m / %Y')}</td>
        </tr>
        <tr>
          <td><b>Degree / Branch:</b>&nbsp;B.E / B.Tech &ndash; All Branches</td>
          <td style="text-align:right;"><b>Semester:</b>&nbsp;Even / Odd</td>
        </tr>
        <tr>
          <td><b>Duration:</b>&nbsp;3 Hours</td>
          <td style="text-align:right;"><b>Maximum Marks:</b>&nbsp;{total}</td>
        </tr>
      </table>
      <div class="exam-inst">
        <b>Instructions to Candidates:</b><br>
        1. Answer <u>ALL</u> questions in both parts.&emsp;
        2. Part A carries 2 marks each.&emsp;
        3. Part B carries 16 marks each.&emsp;
        4. Assume suitable data wherever necessary.&emsp;
        5. Non-programmable calculators permitted where applicable.
      </div>
      <div class="part-header">PART — A &nbsp;({len(pa)} × 2 = {len(pa)*2} Marks)</div>
      <div class="part-sub">Answer ALL {len(pa)} questions. Each question carries 2 marks.</div>
    """, unsafe_allow_html=True)

    for q in pa:
        st.markdown(f"""
        <div class="exam-q-row">
          <div class="exam-q-num">{q['num']}.</div>
          <div><div class="exam-q-text">{q['question']}</div></div>
          <div class="exam-q-mark">({q['marks']})</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
      <br>
      <div class="part-header">PART — B &nbsp;({len(pb)} × 16 = {len(pb)*16} Marks)</div>
      <div class="part-sub">Answer ALL {len(pb)} questions. Each question carries 16 marks.</div>
    """, unsafe_allow_html=True)

    for q in pb:
        st.markdown(f"""
        <div class="exam-q-row">
          <div class="exam-q-num">{q['num']}.</div>
          <div><div class="exam-q-text" style="font-weight:600;">{q['question']}</div></div>
          <div class="exam-q-mark">({q['marks']})</div>
        </div>
        <div style="height:72px;border-bottom:1px solid #ccc;margin-bottom:28px;display:flex;align-items:flex-end;padding-bottom:4px;">
          <span style="font-size:11px;color:#aaa;font-style:italic;">Answer space</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
      <div style="text-align:center;margin-top:48px;padding-top:20px;border-top:2.5px double #111;color:#555;font-size:13px;font-style:italic;letter-spacing:1px;">
        ★★★ End of Question Paper ★★★
      </div>
    </div>""", unsafe_allow_html=True)


# =============================================================================
# QUIZ RESULTS RENDERER
# =============================================================================

def render_quiz_results(responses: dict, subject: str, api_key: str,
                        course_prog: int, t: dict) -> None:
    correct = sum(1 for r in responses.values() if r.get("correct"))
    total   = len(responses)
    score   = correct / total * 100 if total else 0

    CLR_TEXT   = t["CLR_TEXT"]
    CLR_SUB    = t["CLR_SUB"]
    CLR_BORDER = t["CLR_BORDER"]
    BG_CARD    = t["BG_CARD"]
    BG_CARD2   = t["BG_CARD2"]

    topic_scores = collections.defaultdict(lambda: {"correct": 0, "total": 0})
    for r in responses.values():
        tp = r.get("topic", "General")[:30]
        topic_scores[tp]["total"] += 1
        if r.get("correct"):
            topic_scores[tp]["correct"] += 1

    c1, c2 = st.columns([1, 1])
    with c1:
        grade       = "A+" if score >= 90 else "A" if score >= 80 else "B" if score >= 70 else "C" if score >= 60 else "F"
        grade_color = "#16a34a" if score >= 70 else "#dc2626"
        st.markdown(f"""
        <div class="metric-card indigo" style="text-align:center;padding:36px 24px;">
            <div style="font-size:72px;font-weight:700;color:{grade_color};font-family:'Literata',serif;">{grade}</div>
            <div style="font-size:26px;font-weight:800;color:#4f46e5;">{score:.0f}%</div>
            <div style="font-size:15px;font-weight:600;margin-top:4px;">{correct}/{total} correct</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            gauge={
                "axis": {"range": [0, 100]}, "bar": {"color": "#4f46e5"},
                "bgcolor": BG_CARD2,
                "steps": [
                    {"range": [0, 40],  "color": "#fee2e2"},
                    {"range": [40, 70], "color": "#fef9c3"},
                    {"range": [70, 100],"color": "#dcfce7"},
                ],
            },
            number={"suffix": "%", "font": {"color": CLR_TEXT, "size": 36}},
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=240, margin=dict(t=20, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    if topic_scores:
        st.markdown("#### 📊 Topic-by-Topic Breakdown")
        weak_topics = []
        cols_tb = st.columns(min(3, len(topic_scores)))
        for i, (topic, sc) in enumerate(topic_scores.items()):
            pct   = sc["correct"] / sc["total"] * 100 if sc["total"] else 0
            col   = cols_tb[i % len(cols_tb)]
            color = "#16a34a" if pct >= 70 else "#d97706" if pct >= 40 else "#dc2626"
            col.markdown(f"""
            <div class="metric-card" style="border-top:3px solid {color};text-align:center;">
                <div style="font-size:22px;font-weight:800;color:{color};">{pct:.0f}%</div>
                <div style="font-size:12px;margin-top:4px;">{topic}</div>
                <div style="font-size:11px;color:{CLR_SUB};">{sc['correct']}/{sc['total']}</div>
            </div>""", unsafe_allow_html=True)
            if pct < 60:
                weak_topics.append(topic)

        if weak_topics:
            st.markdown("#### 🔴 Weak Areas Detected")
            for wt in weak_topics:
                st.markdown(f"""
                <div class="rec-card rec-high">
                    <div style="font-weight:700;font-size:14px;">⚠️ Revise: {wt}</div>
                    <div style="font-size:13px;color:{CLR_SUB};margin-top:4px;">Score below 60%. Study this topic further.</div>
                    <div style="margin-top:8px;">
                        <a href="https://en.wikipedia.org/wiki/Special:Search?search={wt.replace(' ','+')}" target="_blank" style="color:#4f46e5;font-size:13px;font-weight:600;text-decoration:none;">📖 Wikipedia →</a>
                        &nbsp;&nbsp;
                        <a href="https://www.youtube.com/results?search_query={wt.replace(' ','+')}" target="_blank" style="color:#dc2626;font-size:13px;font-weight:600;text-decoration:none;">▶ YouTube →</a>
                    </div>
                </div>""", unsafe_allow_html=True)

    if score >= 70:
        st.success(f"🏆 Great work! You passed with {score:.0f}%.")
    else:
        st.warning(f"📚 Score is {score:.0f}%. Review weak topics and retry.")

    with st.expander("📋 Full Answer Review"):
        for qn, r in responses.items():
            ok  = r.get("correct", False)
            ic  = "✅" if ok else "❌"
            bg  = "#f0fdf4" if ok else "#fff1f2"
            br  = "#86efac" if ok else "#fca5a5"
            exp = r.get("explanation", "")
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {br};border-radius:10px;padding:12px 18px;margin-bottom:8px;font-size:13.5px;">
                <div style="font-weight:600;">{ic} Q{qn}: <span style="font-weight:400;">{r.get('topic','')[:60]}</span>
                <span class="badge badge-{r.get('difficulty','Medium').lower()}" style="margin-left:8px;">{r.get('difficulty','Medium')}</span></div>
                <div style="margin-top:6px;font-size:12.5px;">Your answer: <code style="background:rgba(0,0,0,.06);padding:2px 7px;border-radius:4px;">{r.get('answer','N/A')}</code></div>
                {f'<div style="margin-top:4px;font-size:12px;color:#555;">💡 {exp[:120]}</div>' if exp else ''}
            </div>""", unsafe_allow_html=True)


# =============================================================================
# COURSE WORKSPACE RENDERER
# =============================================================================

def render_course_workspace(code: str, ctx: str, t: dict, api_key: str = "") -> None:
    c       = next(x for x in COURSES if x["code"] == code)
    syl     = COURSE_SYLLABUS[code]
    lp      = st.session_state["lesson_progress"][code]
    prog    = _course_progress_pct(code)
    total_l = len(lp)
    done_l  = sum(lp.values())

    CLR_TEXT   = t["CLR_TEXT"]
    CLR_SUB    = t["CLR_SUB"]
    CLR_BORDER = t["CLR_BORDER"]
    BG_CARD    = t["BG_CARD"]
    CLR_SOFT   = t["CLR_SOFT"]

    next_lesson = None
    for mod in syl["modules"]:
        for les in mod["lessons"]:
            if not lp.get(les["id"], False):
                next_lesson = les
                break
        if next_lesson:
            break

    if st.button("← Back to Courses", key=f"back_{ctx}_{code}"):
        st.session_state["active_course"] = None
        st.rerun()

    st.markdown(f"""
    <div style="background:{c['bg']};padding:38px 44px;border-radius:18px;color:#fff;margin-bottom:28px;box-shadow:0 6px 28px rgba(0,0,0,.18);">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px;">
            <div>
                <div style="font-size:11px;font-weight:700;letter-spacing:2px;opacity:.8;margin-bottom:6px;">{c['term'].upper()}</div>
                <h1 style="color:#fff;margin:0;font-size:1.85rem;font-family:'Literata',serif;">{syl['title']}</h1>
                <p style="color:rgba(255,255,255,.8);margin:8px 0 18px;font-size:14px;">{syl['description']}</p>
                <div style="display:flex;gap:20px;font-size:13px;flex-wrap:wrap;">
                    <span>👨‍🏫 {syl.get('instructor','')}</span>
                    <span>⭐ {syl.get('rating',4.5)}/5.0</span>
                    <span>👥 {syl.get('enrolled',0):,} enrolled</span>
                    <span>📚 {len(syl['modules'])} Modules · {total_l} Lessons</span>
                </div>
            </div>
            <div style="background:rgba(255,255,255,.18);border-radius:14px;padding:18px 24px;text-align:center;min-width:130px;">
                <div style="font-size:2.4rem;font-weight:800;">{prog}%</div>
                <div style="font-size:12px;opacity:.85;margin-top:2px;">{done_l}/{total_l} Done</div>
                <div style="background:rgba(255,255,255,.28);height:6px;border-radius:3px;overflow:hidden;margin-top:10px;">
                    <div style="width:{prog}%;background:#fff;height:100%;border-radius:3px;"></div>
                </div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    if prog == 100:
        st.success("🏆 Course Complete! You've mastered all lessons.")
    elif next_lesson:
        col_cl, _ = st.columns([2, 3])
        with col_cl:
            st.markdown(f"""
            <div style="background:{BG_CARD};border:2px solid #4f46e5;border-radius:12px;padding:16px 20px;margin-bottom:20px;">
                <div style="font-size:11px;color:#4f46e5;font-weight:700;letter-spacing:1px;margin-bottom:4px;">▶ CONTINUE WHERE YOU LEFT OFF</div>
                <div style="font-weight:700;font-size:14px;">{next_lesson['title']}</div>
                <div style="font-size:12px;color:{CLR_SUB};margin-top:3px;">⏱ {next_lesson['duration']} · {next_lesson['type'].title()}</div>
            </div>""", unsafe_allow_html=True)
            if st.button("Continue Learning →", type="primary", key=f"cont_{ctx}_{code}", use_container_width=True):
                lp[next_lesson["id"]] = True
                st.session_state["global_xp"] += 10
                st.toast(f"✅ '{next_lesson['title']}' marked complete! +10 XP")
                st.rerun()

    st.markdown(f"<h3 style='margin-bottom:16px;color:{CLR_TEXT};'>📋 Course Curriculum</h3>", unsafe_allow_html=True)

    for mod in syl["modules"]:
        mod_lessons = mod["lessons"]
        mod_done    = sum(1 for l in mod_lessons if lp.get(l["id"], False))
        mod_total   = len(mod_lessons)
        mod_pct     = int(mod_done / mod_total * 100)
        mod_icon    = "✅" if mod_done == mod_total else ("🔓" if mod_done > 0 else "🔒")
        expand_mod  = mod_done < mod_total and mod["id"] <= 2

        with st.expander(
            f"{mod_icon}  Module {mod['id']}: {mod['title']}  ·  {mod_done}/{mod_total} lessons · {mod_pct}%",
            expanded=expand_mod,
        ):
            st.markdown(f"""
            <div style="background:{CLR_SOFT};height:5px;border-radius:3px;overflow:hidden;margin-bottom:14px;">
                <div style="width:{mod_pct}%;height:100%;background:linear-gradient(90deg,#4f46e5,#818cf8);border-radius:3px;"></div>
            </div>""", unsafe_allow_html=True)

            for les in mod_lessons:
                done   = lp.get(les["id"], False)
                active = not done and les == next_lesson
                type_icon = {"video": "🎬", "reading": "📖", "quiz": "📝", "project": "🛠"}.get(les["type"], "📄")
                row_cls   = "lesson-done" if done else ("lesson-active" if active else "")
                status_html = (
                    '<span style="background:#dcfce7;color:#15803d;font-size:11px;font-weight:700;padding:2px 10px;border-radius:20px;border:1px solid #86efac;">✓ Done</span>'
                    if done else
                    '<span style="background:#dbeafe;color:#1d4ed8;font-size:11px;font-weight:700;padding:2px 10px;border-radius:20px;border:1px solid #93c5fd;">▶ Up Next</span>'
                    if active else ""
                )
                st.markdown(f"""
                <div class="lesson-row {row_cls}">
                    <span style="font-size:20px;">{type_icon}</span>
                    <div style="flex:1;">
                        <div style="font-weight:600;font-size:13.5px;">{les['id']} — {les['title']}</div>
                        <div style="font-size:11px;color:{CLR_SUB};margin-top:2px;">⏱ {les['duration']} · {les['type'].title()}</div>
                    </div>
                    <div>{status_html}</div>
                </div>""", unsafe_allow_html=True)

                if st.toggle("View notes & links", key=f"toggle_{ctx}_{code}_{les['id']}", value=False):
                    st.markdown(
                        f'<div class="note-box"><strong>📝 Study Notes:</strong><br>{les.get("notes", "No notes available.")}</div>',
                        unsafe_allow_html=True,
                    )
                    lc1, lc2, lc3 = st.columns(3)
                    with lc1:
                        yt = les.get("youtube", "#")
                        st.markdown(f'<a href="{yt}" target="_blank" style="display:inline-block;background:#fee2e2;color:#dc2626;border:1px solid #fca5a5;padding:6px 14px;border-radius:8px;font-size:13px;font-weight:600;text-decoration:none;">▶ YouTube</a>', unsafe_allow_html=True)
                    with lc2:
                        wi = les.get("wiki", "#")
                        st.markdown(f'<a href="{wi}" target="_blank" style="display:inline-block;background:#ede9fe;color:#5b21b6;border:1px solid #c4b5fd;padding:6px 14px;border-radius:8px;font-size:13px;font-weight:600;text-decoration:none;">📖 Wikipedia</a>', unsafe_allow_html=True)
                    with lc3:
                        if not done:
                            if st.button("✅ Mark Done (+10 XP)", key=f"done_{ctx}_{code}_{les['id']}"):
                                lp[les["id"]] = True
                                st.session_state["global_xp"] += 10
                                st.toast("Lesson complete! +10 XP")
                                st.rerun()

            all_done_mod = all(lp.get(l["id"], False) for l in mod_lessons)
            if not all_done_mod:
                st.markdown("---")
                if st.button(
                    f"✅ Mark All Module {mod['id']} Lessons Complete (+{mod_total * 10} XP)",
                    key=f"modall_{ctx}_{code}_{mod['id']}", use_container_width=True,
                ):
                    for l in mod_lessons:
                        if not lp.get(l["id"], False):
                            lp[l["id"]] = True
                            st.session_state["global_xp"] += 10
                    st.toast(f"Module {mod['id']} complete! +{mod_total * 10} XP 🎯")
                    if _course_progress_pct(code) == 100:
                        st.balloons()
                    st.rerun()


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _course_progress_pct(code: str) -> int:
    lp = st.session_state["lesson_progress"][code]
    return int(sum(lp.values()) / len(lp) * 100) if lp else 0
