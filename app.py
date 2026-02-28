# =============================================================================
# app.py  –  CogniLearn Enterprise  |  AI-Based Cognitive Learning Pattern Analyzer
#            Session-state bootstrap, authentication, sidebar, and the two
#            top-level modes: Student Dashboard & AI Analytics Portal.
# =============================================================================

import collections
import datetime
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Local modules
# ---------------------------------------------------------------------------
from config.data import (
    BRAIN_EX, COURSES, COURSE_SYLLABUS, DIFFICULTY_SETTINGS,
    SCHOOLS_PROGRAMS, USERS_DB,
)
from core.engine import (
    ai_generate_exam, ai_generate_quiz, ai_recommendations,
    ai_summarize, extract_text, local_generate_exam,
    local_generate_quiz, local_generate_summary,
    compute_behavioral_features, classify_cognitive_pattern,
    generate_adaptive_recommendations, generate_improvement_report,
    COGNITIVE_PATTERNS,
)
from ui.components import (
    build_theme, inject_css,
    render_course_workspace, render_exam_paper,
    render_quiz, render_quiz_results, render_summary,
    render_behavioral_analytics, render_adaptive_recommendations,
    render_improvement_report,
    _course_progress_pct,
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="CogniLearn — Cognitive Pattern Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SESSION-STATE DEFAULTS
# =============================================================================

DEFAULTS = {
    "logged_in": False, "current_user": None, "synth_data": None,
    "active_training": None, "active_course": None, "global_xp": 0,
    "global_level": 1, "quiz_history": [], "lesson_notes": {},
    "study_plan": {}, "streak_days": [], "last_login_date": None,
    "dark_mode": False,
    "event_log": [],
    "starred_quizzes": [],
    "section_time": {},
    "section_enter_ts": {},
    "quiz_retry_count": {},
    "quiz_start_ts": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

TS_DEF = lambda: {
    "processed": False, "assessment_ready": False, "quiz_data": None,
    "exam_data": None, "exam_type": None, "summary_data": None,
    "file_texts": [], "quiz_responses": {}, "submitted": False,
    "time_limit": 20, "difficulty": "Medium", "quiz_start_ts": None,
}
if "training_state" not in st.session_state:
    st.session_state["training_state"] = TS_DEF()

if "lesson_progress" not in st.session_state:
    lp = {}
    for c in COURSES:
        syl = COURSE_SYLLABUS[c["code"]]
        lp[c["code"]] = {}
        for mod in syl["modules"]:
            for les in mod["lessons"]:
                lp[c["code"]][les["id"]] = False
    st.session_state["lesson_progress"] = lp

if "last_lesson" not in st.session_state:
    st.session_state["last_lesson"] = {c["code"]: None for c in COURSES}

if "users_db" not in st.session_state:
    st.session_state["users_db"] = dict(USERS_DB)

# =============================================================================
# THEME
# =============================================================================

T = build_theme(st.session_state["dark_mode"])
inject_css(T)

BG_MAIN    = T["BG_MAIN"]
BG_CARD    = T["BG_CARD"]
BG_CARD2   = T["BG_CARD2"]
CLR_TEXT   = T["CLR_TEXT"]
CLR_SUB    = T["CLR_SUB"]
CLR_BORDER = T["CLR_BORDER"]
CLR_SOFT   = T["CLR_SOFT"]

# =============================================================================
# EVENT LOGGING
# =============================================================================

def log_event(event: str, detail: str = "") -> None:
    st.session_state["event_log"].append({
        "ts":     datetime.datetime.now().isoformat(timespec="seconds"),
        "event":  event,
        "detail": detail,
    })

def enter_section(name: str) -> None:
    st.session_state["section_enter_ts"][name] = datetime.datetime.now()
    log_event("enter_section", name)

def leave_section(name: str) -> None:
    enter = st.session_state["section_enter_ts"].get(name)
    if enter:
        elapsed = (datetime.datetime.now() - enter).total_seconds()
        prev    = st.session_state["section_time"].get(name, 0)
        st.session_state["section_time"][name] = prev + elapsed
        st.session_state["section_enter_ts"].pop(name, None)

# =============================================================================
# STREAK TRACKING
# =============================================================================

today = datetime.date.today().isoformat()
if st.session_state["last_login_date"] != today:
    st.session_state["last_login_date"] = today
    if today not in st.session_state["streak_days"]:
        st.session_state["streak_days"].append(today)

# =============================================================================
# HELPERS
# =============================================================================

def course_progress_pct(code: str) -> int:
    return _course_progress_pct(code)

def sync_rate() -> int:
    return int(sum(course_progress_pct(c["code"]) for c in COURSES) / len(COURSES))

def reset_training() -> None:
    st.session_state["active_training"] = None
    st.session_state["training_state"]  = TS_DEF()

def reset_course() -> None:
    st.session_state["active_course"] = None

# =============================================================================
# AUTHENTICATION
# =============================================================================

if not st.session_state["logged_in"]:
    st.markdown("""
    <div style="text-align:center;padding:70px 0 24px;">
        <div style="font-size:54px;margin-bottom:14px;">🧠</div>
        <h1 class="grad" style="font-size:3rem;margin:0;">CogniLearn Enterprise</h1>
        <p style="color:#6b6860;font-size:1.1rem;margin-top:10px;">AI-Based Cognitive Learning Pattern Analyzer</p>
        <p style="color:#9090a8;font-size:0.9rem;">Behavioral Analysis · Pattern Classification · Adaptive Recommendations</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        ti, tu = st.tabs(["🔐 Sign In", "📝 Register"])
        with ti:
            with st.form("login"):
                lid  = st.text_input("Registration Number", placeholder="43611162")
                lpwd = st.text_input("Password", type="password", placeholder="••••••")
                if st.form_submit_button("Access Portal →", use_container_width=True, type="primary"):
                    db = st.session_state["users_db"]
                    if lid in db and db[lid]["password"] == lpwd:
                        st.session_state["logged_in"]    = True
                        st.session_state["current_user"] = {**db[lid], "id": lid}
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Demo: 43611162 / pass")
        with tu:
            rid  = st.text_input("Registration No. *")
            rn   = st.text_input("Full Name *")
            rpwd = st.text_input("Password *", type="password")
            ca, cb = st.columns(2)
            with ca:
                rdeg = st.selectbox("Degree", ["Undergraduate", "Postgraduate"])
            with cb:
                ryr  = st.selectbox("Year", ["1st Year", "2nd Year", "3rd Year", "4th Year"])
            rsc  = st.selectbox("School", list(SCHOOLS_PROGRAMS.keys()))
            rpr  = st.selectbox("Program", SCHOOLS_PROGRAMS[rsc])
            if st.button("Create Account", use_container_width=True, type="primary"):
                if rid and rn and rpwd:
                    st.session_state["users_db"][rid] = {
                        "name": rn, "password": rpwd, "degree": rdeg, "year": ryr,
                        "school": rsc, "program": rpr,
                    }
                    st.success("Account created! Please sign in.")
                else:
                    st.error("Fill all required fields.")
    st.stop()

if st.session_state["current_user"] is None:
    st.session_state["logged_in"] = False
    st.rerun()
    st.stop()

# =============================================================================
# SIDEBAR
# =============================================================================

user       = st.session_state.get("current_user") or {}
user_name  = (user.get("name") or "").strip()
user_first = user_name.split()[0] if user_name else "there"
user_id    = user.get("id", "")

with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 0 10px;">
        <div style="font-size:26px;margin-bottom:8px;">🧑‍🎓</div>
        <div style="font-weight:800;font-size:17px;color:#fff;">{user_name}</div>
        <div style="font-size:12px;color:#9090a8;margin-top:3px;">Reg: {user_id}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#2a2a3a;border-radius:10px;padding:12px;margin:8px 0;border:1px solid #3a3a4e;">
        <div style="font-size:12px;color:#a5b4fc;font-weight:600;">{user.get('degree','')} · {user.get('year','')}</div>
        <div style="font-size:12px;color:#9090a8;margin-top:2px;">{user.get('school','')}</div>
        <div style="font-size:13px;color:#d0d0e0;font-weight:600;margin-top:4px;">{user.get('program','')}</div>
    </div>""", unsafe_allow_html=True)

    streak = len(st.session_state["streak_days"])
    st.markdown(
        f'<div class="streak-box" style="margin:8px 0;">🔥 {streak} day streak</div>',
        unsafe_allow_html=True,
    )

    # Live cognitive pattern badge in sidebar
    qhist    = st.session_state.get("quiz_history", [])
    retry_mp = st.session_state.get("quiz_retry_count", {})
    if qhist:
        feats   = compute_behavioral_features(qhist, retry_mp)
        pattern = classify_cognitive_pattern(feats)
        pinfo   = COGNITIVE_PATTERNS.get(pattern, {})
        pc      = pinfo.get("color", "#4f46e5")
        picon   = pinfo.get("icon",  "🧠")
        st.markdown(
            f'<div style="background:{pc}22;border:1px solid {pc};border-radius:10px;'
            f'padding:8px 14px;margin:8px 0;font-size:12px;font-weight:700;color:{pc};">'
            f'{picon} {pattern}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    api_key = st.text_input(
        "🔑 Anthropic API Key", type="password", placeholder="sk-ant-…",
        help="Optional: enables Claude AI quiz/exam generation. Falls back to local NLP without it.",
    )
    if api_key:
        st.markdown('<div style="color:#86efac;font-size:12px;margin-top:-8px;margin-bottom:8px;">✓ Claude AI enabled</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#fbbf24;font-size:12px;margin-top:-8px;margin-bottom:8px;">📝 Using local NLP (no key needed)</div>', unsafe_allow_html=True)

    st.divider()

    dm_label = "☀️ Light Mode" if T["DM"] else "🌙 Dark Mode"
    if st.button(dm_label, use_container_width=True):
        st.session_state["dark_mode"] = not st.session_state["dark_mode"]
        st.rerun()

    app_mode = st.radio(
        "Navigation",
        ["🎓 Student Dashboard", "🤖 AI Analytics Portal"],
        on_change=lambda: (reset_training(), reset_course()),
    )
    st.divider()

    if "Student" in app_mode:
        st.session_state["global_level"] = (st.session_state["global_xp"] // 1000) + 1
        mx     = st.session_state["global_level"] * 1000
        xp_pct = min(st.session_state["global_xp"] / mx, 1.0)
        st.markdown(f"""
        <div style="margin-bottom:10px;">
            <div style="font-weight:700;color:#a5b4fc;font-size:14px;">⚡ Level {st.session_state['global_level']} Pioneer</div>
            <div style="font-size:12px;color:#9090a8;margin:3px 0;">{st.session_state['global_xp']} / {mx} XP</div>
            <div style="background:#2a2a3a;height:7px;border-radius:4px;overflow:hidden;margin-top:4px;">
                <div style="width:{xp_pct*100}%;height:100%;background:linear-gradient(90deg,#6366f1,#a78bfa);border-radius:4px;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    if st.button("Sign Out", use_container_width=True):
        st.session_state["logged_in"] = False
        st.rerun()

# =============================================================================
# MODE 1 – STUDENT DASHBOARD
# =============================================================================

if "Student" in app_mode:
    t_hub, t_courses, t_train, t_prog = st.tabs(
        ["🏠 Hub", "📚 Courses", "🧠 Brain Training", "📊 Analytics"]
    )

    # ── HUB ──────────────────────────────────────────────────────────────────
    with t_hub:
        if not st.session_state["active_course"]:
            st.markdown(
                f"<h1 style='margin-bottom:6px;'>Welcome back, <span class='grad'>{user_first}</span></h1>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='color:{CLR_SUB};margin-bottom:26px;'>AI-powered learning with real-time cognitive pattern analysis.</p>",
                unsafe_allow_html=True,
            )

            sr = sync_rate()
            total_lessons = sum(
                len(COURSE_SYLLABUS[c["code"]]["modules"][m]["lessons"])
                for c in COURSES
                for m in range(len(COURSE_SYLLABUS[c["code"]]["modules"]))
            )
            done_lessons = sum(
                sum(lp.values())
                for lp in [st.session_state["lesson_progress"][c["code"]] for c in COURSES]
            )

            # Show cognitive pattern on hub
            qhist_hub = st.session_state.get("quiz_history", [])
            retry_hub = st.session_state.get("quiz_retry_count", {})
            if qhist_hub:
                feats_hub   = compute_behavioral_features(qhist_hub, retry_hub)
                pattern_hub = classify_cognitive_pattern(feats_hub)
                pinfo_hub   = COGNITIVE_PATTERNS.get(pattern_hub, {})
                pc_hub      = pinfo_hub.get("color", "#4f46e5")
                picon_hub   = pinfo_hub.get("icon", "🧠")
                st.markdown(f"""
                <div style="background:{pc_hub}11;border:1.5px solid {pc_hub};border-radius:12px;padding:14px 20px;margin-bottom:20px;display:flex;align-items:center;gap:14px;">
                    <div style="font-size:32px;">{picon_hub}</div>
                    <div>
                        <div style="font-size:11px;font-weight:700;letter-spacing:1.5px;color:{pc_hub};text-transform:uppercase;">Your Cognitive Learning Pattern</div>
                        <div style="font-size:18px;font-weight:800;color:{CLR_TEXT};">{pattern_hub}</div>
                        <div style="font-size:12px;color:{CLR_SUB};">{pinfo_hub.get('desc','')}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            for col, (title, val, cls) in zip([c1, c2, c3, c4], [
                ("Active Courses",   str(len(COURSES)),               "indigo"),
                ("Sync Rate",        f"{sr}%",                         "green"),
                ("Lessons Done",     f"{done_lessons}/{total_lessons}", "amber"),
                ("Total XP",         str(st.session_state["global_xp"]), "rose"),
            ]):
                with col:
                    st.markdown(f"""
                    <div class="metric-card {cls}">
                        <div style="color:{CLR_SUB};font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;">{title}</div>
                        <div style="font-size:2rem;font-weight:800;font-family:'Literata',serif;color:{CLR_TEXT};margin-top:6px;">{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(f"<h3 style='margin:26px 0 14px;color:{CLR_TEXT};'>Recent Courses</h3>", unsafe_allow_html=True)
            cols = st.columns(3)
            for idx, c in enumerate(COURSES[:3]):
                p = course_progress_pct(c["code"])
                with cols[idx]:
                    st.markdown(f"""
                    <div style="border-radius:14px;overflow:hidden;background:{BG_CARD};border:1px solid {CLR_BORDER};box-shadow:0 2px 8px rgba(0,0,0,0.05);margin-bottom:8px;">
                        <div style="height:92px;background:{c['bg']};"></div>
                        <div style="padding:16px;">
                            <div style="color:{CLR_SUB};font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;">{c['term']}</div>
                            <div style="color:{CLR_TEXT};font-size:13px;font-weight:700;margin:5px 0 12px;line-height:1.4;">{c['code']}</div>
                            <div style="background:{CLR_SOFT};height:5px;border-radius:3px;overflow:hidden;">
                                <div style="width:{p}%;height:100%;background:linear-gradient(90deg,#4f46e5,#818cf8);border-radius:3px;"></div>
                            </div>
                            <div style="font-size:11px;color:{CLR_SUB};margin-top:5px;font-weight:600;">{p}% complete</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                    if st.button("Open →", key=f"dash_{c['code']}", use_container_width=True):
                        st.session_state["active_course"] = c["code"]
                        st.rerun()

            st.markdown(f"<h3 style='margin:26px 0 14px;color:{CLR_TEXT};'>All Courses Overview</h3>", unsafe_allow_html=True)
            all_cols = st.columns(3)
            for idx, c in enumerate(COURSES[3:]):
                p = course_progress_pct(c["code"])
                with all_cols[idx]:
                    st.markdown(f"""
                    <div style="border-radius:14px;overflow:hidden;background:{BG_CARD};border:1px solid {CLR_BORDER};margin-bottom:8px;">
                        <div style="height:62px;background:{c['bg']};"></div>
                        <div style="padding:12px 16px;">
                            <div style="color:{CLR_TEXT};font-size:12px;font-weight:700;">{c['code']}</div>
                            <div style="background:{CLR_SOFT};height:4px;border-radius:2px;overflow:hidden;margin:6px 0 4px;">
                                <div style="width:{p}%;height:100%;background:linear-gradient(90deg,#4f46e5,#818cf8);border-radius:2px;"></div>
                            </div>
                            <div style="font-size:11px;color:{CLR_SUB};">{p}%</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                    if st.button("Open →", key=f"dash2_{c['code']}", use_container_width=True):
                        st.session_state["active_course"] = c["code"]
                        st.rerun()
        else:
            render_course_workspace(st.session_state["active_course"], "hub", T, api_key)

    # ── COURSES ───────────────────────────────────────────────────────────────
    with t_courses:
        if not st.session_state["active_course"]:
            st.markdown(f"<h1 style='color:{CLR_TEXT};'>Course Library</h1>", unsafe_allow_html=True)
            cols = st.columns(3)
            for idx, c in enumerate(COURSES):
                p   = course_progress_pct(c["code"])
                syl = COURSE_SYLLABUS[c["code"]]
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div style="border-radius:14px;overflow:hidden;background:{BG_CARD};border:1px solid {CLR_BORDER};box-shadow:0 2px 8px rgba(0,0,0,0.04);margin-bottom:8px;">
                        <div style="height:80px;background:{c['bg']};display:flex;align-items:flex-end;padding:12px;">
                            <div style="font-size:10px;color:rgba(255,255,255,.8);font-weight:700;letter-spacing:1.5px;">{c['term'].upper()}</div>
                        </div>
                        <div style="padding:15px;">
                            <div style="color:#4f46e5;font-size:12px;font-weight:700;">{c['code']}</div>
                            <div style="color:{CLR_TEXT};font-size:13px;font-weight:500;margin:4px 0 6px;line-height:1.4;min-height:36px;">{c['title']}</div>
                            <div style="font-size:11px;color:{CLR_SUB};margin-bottom:8px;">⭐ {syl.get('rating',4.5)} · 👥 {syl.get('enrolled',0):,}</div>
                            <div style="background:{CLR_SOFT};height:5px;border-radius:3px;overflow:hidden;">
                                <div style="width:{p}%;height:100%;background:linear-gradient(90deg,#4f46e5,#818cf8);border-radius:3px;"></div>
                            </div>
                            <div style="font-size:11px;color:{CLR_SUB};margin-top:5px;">{p}% complete</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                    if st.button("Enter Course →", key=f"lib_{c['code']}", use_container_width=True):
                        st.session_state["active_course"] = c["code"]
                        st.rerun()
        else:
            render_course_workspace(st.session_state["active_course"], "lib", T, api_key)

    # ── BRAIN TRAINING ────────────────────────────────────────────────────────
    with t_train:
        ts = st.session_state["training_state"]

        if not st.session_state["active_training"]:
            st.markdown(f"<h1 style='color:{CLR_TEXT};'>Assessment Generator</h1>", unsafe_allow_html=True)
            st.markdown(
                f"<p style='color:{CLR_SUB};margin-bottom:26px;'>Upload your study materials → AI extracts knowledge → Generates custom quizzes & exam papers.<br>"
                "All quiz attempts are analyzed to build your <strong>Cognitive Learning Pattern</strong>.</p>",
                unsafe_allow_html=True,
            )
            cols = st.columns(3)
            for idx, ex in enumerate(BRAIN_EX):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="card" style="text-align:center;min-height:300px;">
                        <div style="width:72px;height:72px;border-radius:18px;background:{ex['bg']};
                                    display:inline-flex;align-items:center;justify-content:center;
                                    font-size:36px;margin-bottom:14px;box-shadow:0 6px 16px rgba(0,0,0,0.12);">{ex['icon']}</div>
                        <h3 style="margin:0 0 4px;font-size:17px;color:{CLR_TEXT};">{ex['title']}</h3>
                        <div style="color:{ex['color']};font-size:13px;font-weight:600;margin-bottom:10px;">({ex['subtitle']})</div>
                        <p style="color:{CLR_SUB};font-size:13px;line-height:1.6;margin-bottom:14px;">{ex['desc']}</p>
                        <div style="background:{CLR_SOFT};border-radius:8px;padding:8px;border:1px solid {CLR_BORDER};">
                            <span style="color:#d97706;font-weight:700;">+{ex['xp']} XP</span>
                            <span style="color:{CLR_SUB};font-size:12px;"> on completion</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                    if st.button("Start Workspace", key=f"start_{ex['id']}", use_container_width=True, type="primary"):
                        st.session_state["active_training"] = ex["id"]
                        st.rerun()
        else:
            ex = next(x for x in BRAIN_EX if x["id"] == st.session_state["active_training"])
            st.button("← Back to Modules", on_click=reset_training)

            st.markdown(f"""
            <div style="background:{ex['bg']};padding:28px 38px;border-radius:16px;color:#fff;margin-bottom:28px;box-shadow:0 4px 18px rgba(0,0,0,0.12);">
                <h2 style="color:#fff;margin:0;font-family:'Literata',serif;">{ex['icon']} {ex['title']} Workspace</h2>
                <div style="color:rgba(255,255,255,.8);font-size:14px;margin-top:6px;">{ex['subtitle']} · +{ex['xp']} XP on completion</div>
                <div style="margin-top:8px;font-size:12px;color:rgba(255,255,255,.7);">
                    {'✨ Claude AI active — high-quality generation' if api_key else '📝 Local NLP mode — no API key needed'} · 🧠 Behavioral data is collected for pattern analysis
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("### 📥 Step 1: Upload Study Materials")
            uploaded_files = st.file_uploader(
                "Upload files (PDF, DOCX, TXT, PPTX)",
                accept_multiple_files=True, type=ex["fmts"], key="brain_up",
            )

            if uploaded_files:
                st.success(f"✓ {len(uploaded_files)} file(s) loaded.")

                if not ts["processed"]:
                    if st.button("🧠 Extract Knowledge", type="primary", use_container_width=True):
                        with st.spinner("Reading and understanding your documents…"):
                            texts = [extract_text(f) for f in uploaded_files]
                            sdata = ai_summarize(texts, ex["subtitle"], api_key) if api_key else None
                            if not sdata:
                                sdata = local_generate_summary(texts)
                        ts["file_texts"]   = texts
                        ts["summary_data"] = sdata
                        ts["processed"]    = True
                        st.rerun()

                if ts["processed"]:
                    st.markdown("### 🧠 Step 2: Knowledge Extraction Results")
                    if ts.get("summary_data"):
                        src_label = "Claude AI" if api_key else "Local NLP"
                        st.markdown(f"""
                        <div style="background:{CLR_SOFT};border-left:5px solid {ex['color']};border-radius:12px;padding:16px 22px;margin-bottom:16px;">
                            <div style="font-weight:700;font-size:16px;color:{ex['color']};">📋 {src_label} Analysis Complete</div>
                            <div style="font-size:13px;color:{CLR_SUB};margin-top:4px;">Extracted from {len(uploaded_files)} file(s)</div>
                        </div>""", unsafe_allow_html=True)
                        render_summary(ts["summary_data"], T, ex["color"])

                    if not ts["assessment_ready"]:
                        st.markdown("### 📝 Step 3: Configure & Generate Assessment")
                        exam_type = st.radio(
                            "Assessment Type:",
                            ["🎯 Custom Quiz (MCQ + Multi-Select + Fill Blanks)",
                             "📄 Model Exam Paper (University Format)"],
                            horizontal=True,
                        )
                        cA, cB, cC = st.columns(3)
                        if "Quiz" in exam_type:
                            with cA: num_q  = st.number_input("Number of Questions", 5, 25, 10, 5)
                            with cB: time_l = st.number_input("Time Limit (mins)", 10, 120, 20, 5)
                            with cC: diff   = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
                        else:
                            with cA: n2m  = st.number_input("Part A – 2-Mark Qs", 5, 20, 10)
                            with cB: n16m = st.number_input("Part B – 16-Mark Qs", 1, 5, 3)
                            with cC: diff = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])

                        if st.button("🚀 Generate Assessment", type="primary", use_container_width=True):
                            with st.spinner("Crafting your personalised assessment…"):
                                ftexts = ts.get("file_texts", [""])
                                if "Quiz" in exam_type:
                                    qdata = ai_generate_quiz(ftexts, ex["subtitle"], int(num_q), diff, api_key) if api_key else None
                                    if not qdata:
                                        qdata = local_generate_quiz(ftexts, int(num_q), diff)
                                    ts["quiz_data"]  = qdata
                                    ts["time_limit"] = int(time_l)
                                    ts["difficulty"] = diff
                                else:
                                    edata = ai_generate_exam(ftexts, ex["subtitle"], int(n2m), int(n16m), api_key) if api_key else None
                                    if not edata:
                                        edata = local_generate_exam(ftexts, ex["subtitle"], int(n2m), int(n16m))
                                    ts["exam_data"] = edata
                                ts["exam_type"]        = exam_type
                                ts["assessment_ready"] = True
                            st.rerun()

                    if ts["assessment_ready"] and not ts.get("submitted"):
                        st.divider()
                        if "Quiz" in ts.get("exam_type", ""):
                            if ts["quiz_data"]:
                                render_quiz(
                                    ts["quiz_data"], ex["id"], ts["time_limit"],
                                    ex["subtitle"], ex["xp"], ts.get("difficulty", "Medium"), T,
                                )
                            else:
                                st.error("Quiz generation failed. Check your files have readable text.")
                        else:
                            if ts["exam_data"]:
                                render_exam_paper(ts["exam_data"], ex["subtitle"], T)
                                st.markdown("<br>", unsafe_allow_html=True)
                                cA2, cB2 = st.columns(2)
                                with cA2:
                                    st.info("💡 Use Ctrl+P / Cmd+P to print / save as PDF.")
                                with cB2:
                                    if st.button(f"✅ Submit & Claim {ex['xp']} XP", type="primary", use_container_width=True):
                                        st.session_state["global_xp"] += ex["xp"]
                                        ts["submitted"] = True
                                        st.rerun()
                            else:
                                st.error("Exam paper generation failed.")

                    if ts.get("submitted"):
                        if ts.get("quiz_responses"):
                            render_quiz_results(ts["quiz_responses"], ex["subtitle"], api_key, sync_rate(), T)
                        else:
                            st.success(f"✅ Exam submitted! +{ex['xp']} XP claimed!")
                            st.balloons()
                        if st.button("← Start New Assessment", use_container_width=True):
                            reset_training()
                            st.rerun()

    # ── ANALYTICS ─────────────────────────────────────────────────────────────
    with t_prog:
        log_event("view_analytics", "")
        st.markdown("<h1 class='grad'>Learning Analytics</h1>", unsafe_allow_html=True)

        total_l_all   = sum(len(st.session_state["lesson_progress"][c["code"]]) for c in COURSES)
        done_l_all    = sum(sum(st.session_state["lesson_progress"][c["code"]].values()) for c in COURSES)
        overall       = int(done_l_all / total_l_all * 100) if total_l_all else 0
        qhist         = st.session_state["quiz_history"]
        avg_q         = float(np.mean([q["score"] for q in qhist])) if qhist else 0.0
        total_correct = sum(q.get("correct", 0) for q in qhist)
        total_wrong   = sum(q.get("wrong",   0) for q in qhist)
        sec_time      = st.session_state.get("section_time", {})
        retry_map     = st.session_state.get("quiz_retry_count", {})
        starred_q     = st.session_state.get("starred_quizzes", [])

        # Compute pattern for this tab
        features_tab = compute_behavioral_features(qhist, retry_map)
        pattern_tab  = classify_cognitive_pattern(features_tab)

        # KPIs
        kpi_cols = st.columns(6)
        kpi_data = [
            ("Overall Progress",  f"{overall}%",                      "indigo"),
            ("Lessons Done",      f"{done_l_all}/{total_l_all}",      "green"),
            ("Avg Quiz Score",    f"{avg_q:.0f}%",                    "amber"),
            ("XP Earned",         str(st.session_state["global_xp"]), "rose"),
            ("Quizzes Taken",     str(len(qhist)),                    "sky"),
            ("Starred Quizzes",   str(len(starred_q)),                "violet"),
        ]
        for col, (title, val, cls) in zip(kpi_cols, kpi_data):
            with col:
                st.markdown(f"""
                <div class="metric-card {cls}">
                    <div style="color:{CLR_SUB};font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;">{title}</div>
                    <div style="font-size:1.7rem;font-weight:800;font-family:'Literata',serif;color:{CLR_TEXT};margin-top:6px;">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        at1, at2, at3, at4, at5, at6 = st.tabs([
            "📊 Overview",
            "🧠 Behavioral Analysis",
            "🎯 Adaptive Recommendations",
            "📈 Improvement Report",
            "🎯 Quiz Deep-Dive",
            "🤖 Aggregated Scores",
        ])

        # ── Tab 1: Overview ──
        with at1:
            col_l, col_r = st.columns([3, 2])
            with col_l:
                st.markdown("#### Course Completion")
                course_df = pd.DataFrame([
                    {"Course": c["code"],
                     "Completed": course_progress_pct(c["code"]),
                     "Remaining": 100 - course_progress_pct(c["code"])}
                    for c in COURSES
                ])
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(y=course_df["Course"], x=course_df["Completed"],
                    orientation="h", name="Completed", marker_color="#4f46e5", marker_line_width=0))
                fig_bar.add_trace(go.Bar(y=course_df["Course"], x=course_df["Remaining"],
                    orientation="h", name="Remaining", marker_color=CLR_SOFT, marker_line_width=0))
                fig_bar.update_layout(barmode="stack", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(l=0, r=10, t=10, b=0),
                    xaxis=dict(range=[0,100], tickfont=dict(color=CLR_SUB), gridcolor=CLR_SOFT, ticksuffix="%"),
                    yaxis=dict(tickfont=dict(color=CLR_TEXT, size=12)),
                    legend=dict(font=dict(color=CLR_SUB)), font=dict(color=CLR_TEXT))
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_r:
                st.markdown("#### Aptitude Radar")
                cats   = ["Memory", "Analysis", "Focus", "Speed", "Application"]
                base   = max(10, avg_q)
                r_vals = [min(100, base*0.9), min(100, overall), min(100, base*0.95),
                          min(100, 55+len(qhist)*2), min(100, overall*0.8+10)]
                r_vals = [max(8, v) for v in r_vals]
                fig_r  = go.Figure(go.Scatterpolar(
                    r=r_vals+[r_vals[0]], theta=cats+[cats[0]],
                    fill="toself", fillcolor="rgba(79,70,229,0.18)",
                    line=dict(color="#4f46e5", width=2.5),
                    marker=dict(color="#4f46e5", size=8)))
                fig_r.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0,100],
                                   tickfont=dict(color=CLR_SUB, size=9), gridcolor=CLR_BORDER),
                               angularaxis=dict(tickfont=dict(color=CLR_TEXT, size=11), gridcolor=CLR_BORDER),
                               bgcolor="rgba(0,0,0,0)"),
                    paper_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(t=20,b=20,l=30,r=30))
                st.plotly_chart(fig_r, use_container_width=True)

        # ── Tab 2: BEHAVIORAL ANALYSIS (NEW CORE FEATURE) ──
        with at2:
            render_behavioral_analytics(qhist, retry_map, overall, T)

        # ── Tab 3: ADAPTIVE RECOMMENDATIONS (NEW CORE FEATURE) ──
        with at3:
            render_adaptive_recommendations(qhist, retry_map, overall, api_key, T)

        # ── Tab 4: IMPROVEMENT REPORT (NEW CORE FEATURE) ──
        with at4:
            render_improvement_report(qhist, retry_map, overall, T)

        # ── Tab 5: Quiz Deep-Dive ──
        with at5:
            if not qhist:
                st.info("📊 Complete a quiz in Brain Training to unlock performance analytics.")
            else:
                st.markdown("#### 📈 Quiz Score Timeline")
                df_q = pd.DataFrame([
                    {"Attempt": i+1, "Subject": q["subject"][:22],
                     "Score": q["score"], "Correct": q["correct"],
                     "Wrong": q.get("wrong", q["total"]-q["correct"]),
                     "Total": q["total"],
                     "Difficulty": q.get("difficulty", "Medium"),
                     "Duration (min)": round(q.get("duration_s", 0)/60, 1)}
                    for i, q in enumerate(qhist)
                ])
                fig_l = go.Figure()
                fig_l.add_trace(go.Scatter(
                    x=df_q["Attempt"], y=df_q["Score"],
                    mode="lines+markers+text", name="Score",
                    line=dict(color="#4f46e5", width=3),
                    marker=dict(size=11, color="#4f46e5", line=dict(color="#fff", width=2)),
                    text=[f"{s:.0f}%" for s in df_q["Score"]],
                    textposition="top center", textfont=dict(color=CLR_TEXT, size=12),
                    fill="tozeroy", fillcolor="rgba(79,70,229,0.08)"))
                fig_l.add_hline(y=70, line_dash="dash", line_color="#f59e0b",
                                annotation_text="Pass line (70%)", annotation_font_color="#d97706")
                fig_l.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=320, margin=dict(l=0,r=50,t=20,b=10),
                    xaxis=dict(title="Attempt #", tickfont=dict(color=CLR_SUB), gridcolor=CLR_SOFT),
                    yaxis=dict(title="Score %", range=[0,105], tickfont=dict(color=CLR_SUB), gridcolor=CLR_SOFT),
                    legend=dict(font=dict(color=CLR_SUB)), font=dict(color=CLR_TEXT))
                st.plotly_chart(fig_l, use_container_width=True)

                with st.expander("📋 Detailed Quiz Log"):
                    st.dataframe(df_q, use_container_width=True, height=240)

        # ── Tab 6: Aggregated Scores ──
        with at6:
            st.markdown("#### 🤖 Aggregated Score Analysis with StandardScaler Normalisation")
            if not qhist:
                st.info("Complete at least one quiz to see aggregated score analysis.")
            else:
                raw_rows = [
                    {"Attempt": i+1, "Subject": q["subject"][:22],
                     "Score": q["score"], "Correct": q.get("correct",0),
                     "Wrong": q.get("wrong", q["total"]-q.get("correct",0)),
                     "Total_Qs": q["total"], "XP_Earned": q.get("xp",0),
                     "Duration_s": q.get("duration_s",0),
                     "Difficulty": q.get("difficulty","Medium")}
                    for i, q in enumerate(qhist)
                ]
                df_raw   = pd.DataFrame(raw_rows)
                num_feat = ["Score","Correct","Wrong","Total_Qs","XP_Earned","Duration_s"]
                df_num   = df_raw[num_feat].copy()

                if len(df_num) >= 2:
                    scaler     = StandardScaler()
                    scaled_arr = scaler.fit_transform(df_num)
                    df_scaled  = pd.DataFrame(scaled_arr, columns=[f"Z_{c}" for c in num_feat])

                    agg1, agg2 = st.columns(2)
                    with agg1:
                        st.markdown("**Raw Score Statistics**")
                        st.dataframe(df_num.describe().round(2).T, use_container_width=True, height=240)
                    with agg2:
                        st.markdown("**StandardScaler Z-Score Statistics**")
                        st.dataframe(df_scaled.describe().round(3).T, use_container_width=True, height=240)

                    st.markdown("#### Z-Score Heatmap (per Attempt)")
                    df_heat       = df_scaled.copy()
                    df_heat.index = [f"A{i+1}" for i in range(len(df_heat))]
                    fig_heat = go.Figure(go.Heatmap(
                        z=df_heat.values.tolist(), x=df_heat.columns.tolist(), y=df_heat.index.tolist(),
                        colorscale="RdYlGn", zmid=0,
                        text=[[f"{v:.2f}" for v in row] for row in df_heat.values],
                        texttemplate="%{text}", textfont=dict(size=10), showscale=True))
                    fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                        height=max(220, len(df_heat)*38), margin=dict(l=40,r=20,t=20,b=40),
                        xaxis=dict(tickfont=dict(color=CLR_TEXT,size=11)),
                        yaxis=dict(tickfont=dict(color=CLR_TEXT,size=11)),
                        font=dict(color=CLR_TEXT))
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("Complete at least 2 quizzes for statistical analysis.")

# =============================================================================
# MODE 2 – AI ANALYTICS PORTAL
# =============================================================================

elif "Analytics" in app_mode:
    st.markdown("<h1 class='grad' style='text-align:center;'>🤖 AI Research Laboratory</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;color:{CLR_SUB};margin-bottom:28px;'>Upload a student dataset or synthesise one to explore ML-driven cognitive pattern analytics.</p>", unsafe_allow_html=True)

    cU, cS = st.columns(2)
    with cU:
        up_file = st.file_uploader("Upload Student Dataset (CSV)", type=["csv"])
    with cS:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.write("No CSV? Generate a synthetic dataset with behavioral features:")
        if st.button("🧬 Synthesise Enterprise Dataset (5 000 Records)", use_container_width=True, type="primary"):
            with st.spinner("Generating…"):
                np.random.seed(42); n = 5000
                sh     = np.random.normal(15, 5, n).clip(2, 35)
                sl     = np.random.normal(7, 1.5, n).clip(4, 10)
                stress = np.random.uniform(1, 10, n)
                pr     = np.random.uniform(40, 100, n)
                # Response time: inverse of study hours (more study = faster)
                resp_t = np.random.normal(60, 20, n).clip(15, 180) - sh * 1.5
                resp_t = resp_t.clip(15, 150)
                # Mistake frequency
                mistake_f = np.random.beta(2, 5, n)
                mistake_f = (mistake_f + stress * 0.03).clip(0.05, 0.95)
                # Retry rate
                retry_r = np.random.poisson(1.5, n).clip(1, 6).astype(float)
                sc = (70 + sh*2 - stress*2.5 + sl*1.5 + np.random.normal(0, 8, n)).clip(0, 100)
                df = pd.DataFrame({
                    "Student_ID":         [f"S{1000+i}" for i in range(n)],
                    "Study_Hours":        sh.round(1),
                    "Sleep_Hours":        sl.round(1),
                    "Stress_Level":       stress.round(1),
                    "Participation_Rate": pr.round(1),
                    "Avg_Response_Time_s":resp_t.round(1),
                    "Mistake_Frequency":  mistake_f.round(3),
                    "Retry_Rate":         retry_r,
                    "Assessment_Score":   sc.round(1),
                })
                st.session_state["synth_data"] = df
            st.success("✓ 5 000 records generated with behavioral features.")

    active_df = None
    if up_file:
        active_df = pd.read_csv(up_file)
    elif st.session_state.get("synth_data") is not None:
        active_df = st.session_state["synth_data"].copy()

    if active_df is not None:

        @st.cache_data
        def run_ml(df_in: pd.DataFrame):
            df = df_in.copy()
            if "Student_ID" not in df.columns:
                df.insert(0, "Student_ID", [f"STU-{i:04d}" for i in range(len(df))])
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols) < 3:
                return df, num_cols, False
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
            X   = StandardScaler().fit_transform(df[num_cols])
            km  = KMeans(n_clusters=4, random_state=42, n_init=10)
            df["Cluster"] = km.fit_predict(X)
            pca_arr = PCA(n_components=3).fit_transform(X)
            df[["PCA_1","PCA_2","PCA_3"]] = pca_arr
            order  = df.groupby("Cluster")[num_cols[-1]].mean().sort_values().index
            labels = {
                order[0]: "Foundational Learner",
                order[1]: "Impulsive Responder",
                order[2]: "Analytical Thinker",
                order[3]: "Rapid Master",
            }
            df["Cognitive_Pattern"] = df["Cluster"].map(labels)
            return df, num_cols, True

        df_ml, n_cols, ok = run_ml(active_df)
        if not ok:
            st.error("Dataset needs at least 3 numeric columns.")
            st.stop()

        PAL = {
            "Foundational Learner": "#ef4444",
            "Impulsive Responder":  "#f59e0b",
            "Analytical Thinker":   "#3b82f6",
            "Rapid Master":         "#22c55e",
        }

        t1, t2, t3, t4 = st.tabs(["📋 Data Matrix", "🧬 EDA Analysis", "🌌 3D Clusters", "🤖 ML Engine"])

        with t1:
            st.markdown("### Student Dataset Overview")
            counts = df_ml["Cognitive_Pattern"].value_counts()
            cc = st.columns(4)
            for col, (pat, cnt) in zip(cc, counts.items()):
                pct = cnt / len(df_ml) * 100
                col.markdown(f"""
                <div class="metric-card" style="text-align:center;border-top:3px solid {PAL[pat]};">
                    <div style="font-size:1.6rem;font-weight:800;font-family:'Literata',serif;color:{PAL[pat]};">{pct:.1f}%</div>
                    <div style="font-size:12.5px;color:{CLR_TEXT};font-weight:600;margin-top:3px;">{pat}</div>
                    <div style="font-size:11px;color:{CLR_SUB};">{cnt:,} students</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Descriptive Statistics")
            desc = df_ml[n_cols].describe().T.round(2)
            desc.columns = [c.title() for c in desc.columns]
            st.dataframe(desc.style.background_gradient(cmap="Blues", subset=["Mean","Std"]), use_container_width=True)
            st.markdown("#### Raw Data (first 100 rows)")
            st.dataframe(df_ml.head(100), use_container_width=True, height=380)

        with t2:
            st.markdown("### Exploratory Data Analysis")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Correlation Heatmap")
                corr  = df_ml[n_cols].corr()
                fig_h = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                  color_continuous_midpoint=0, aspect="auto")
                fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=380, margin=dict(t=10,b=10,l=0,r=0), font=dict(color=CLR_TEXT))
                st.plotly_chart(fig_h, use_container_width=True)
            with c2:
                st.markdown("#### Feature Distribution by Pattern")
                feat  = st.selectbox("Select feature:", n_cols, key="eda_feat")
                fig_v = px.violin(df_ml, x="Cognitive_Pattern", y=feat,
                                  color="Cognitive_Pattern", color_discrete_map=PAL, box=True, points="outliers")
                fig_v.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False, height=380, margin=dict(t=10,b=10,l=0,r=0),
                    xaxis=dict(tickfont=dict(color=CLR_SUB)), yaxis=dict(tickfont=dict(color=CLR_SUB)),
                    font=dict(color=CLR_TEXT))
                st.plotly_chart(fig_v, use_container_width=True)

            st.markdown("#### Pair-wise Scatter Matrix")
            fig_sm = px.scatter_matrix(
                df_ml.sample(min(500, len(df_ml)), random_state=42),
                dimensions=n_cols[:4], color="Cognitive_Pattern",
                color_discrete_map=PAL, opacity=0.6)
            fig_sm.update_traces(marker=dict(size=3))
            fig_sm.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=540,
                margin=dict(t=20,b=20,l=20,r=20), font=dict(color=CLR_TEXT, size=11))
            st.plotly_chart(fig_sm, use_container_width=True)

            cA, cB = st.columns(2)
            with cA:
                st.markdown("#### Feature Interaction Network")
                G = nx.Graph()
                for col in n_cols: G.add_node(col)
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        w = corr.iloc[i, j]
                        if abs(w) > 0.15:
                            G.add_edge(corr.columns[i], corr.columns[j], weight=w)
                fig_nx, ax_nx = plt.subplots(figsize=(7, 5))
                nx_bg = "#252538" if T["DM"] else "#faf9f5"
                fig_nx.patch.set_facecolor(nx_bg); ax_nx.set_facecolor(nx_bg); ax_nx.axis("off")
                pos = nx.spring_layout(G, k=1.2, seed=42)
                ec  = ["#16a34a" if G[u][v]["weight"] > 0 else "#dc2626" for u, v in G.edges()]
                ew  = [abs(G[u][v]["weight"]) * 7 for u, v in G.edges()]
                nx.draw_networkx_nodes(G, pos, node_color="#4f46e5", node_size=2200,
                                       edgecolors="#818cf8", linewidths=2, ax=ax_nx)
                nx.draw_networkx_edges(G, pos, width=ew, edge_color=ec, alpha=0.65, ax=ax_nx)
                nx.draw_networkx_labels(G, pos,
                    labels={n: n.replace("_","\n") for n in G.nodes()},
                    font_size=8, font_color="white", font_weight="bold", ax=ax_nx)
                ax_nx.legend(handles=[
                    mpatches.Patch(color="#16a34a", label="Positive"),
                    mpatches.Patch(color="#dc2626", label="Negative"),
                ], loc="lower left", fontsize=9, framealpha=0.8, facecolor=nx_bg, labelcolor=CLR_TEXT)
                st.pyplot(fig_nx, clear_figure=True)

            with cB:
                st.markdown("#### Correlation with Assessment Score")
                tgt      = "Assessment_Score" if "Assessment_Score" in n_cols else n_cols[-1]
                corr_tgt = df_ml[n_cols].corr()[tgt].drop(tgt).sort_values()
                clrs_bar = ["#ef4444" if v < 0 else "#16a34a" for v in corr_tgt.values]
                fig_cb   = go.Figure(go.Bar(
                    y=corr_tgt.index, x=corr_tgt.values,
                    orientation="h", marker_color=clrs_bar, marker_line_width=0))
                fig_cb.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=320, margin=dict(t=10,b=10,l=0,r=0),
                    xaxis=dict(title="Correlation", tickfont=dict(color=CLR_SUB), gridcolor=CLR_SOFT,
                               range=[-1,1], zeroline=True, zerolinecolor=CLR_BORDER, zerolinewidth=2),
                    yaxis=dict(tickfont=dict(color=CLR_TEXT)), font=dict(color=CLR_TEXT))
                st.plotly_chart(fig_cb, use_container_width=True)

        with t3:
            st.markdown("### 3D Cognitive Cluster Space (PCA Projection)")
            sample = df_ml.sample(min(1500, len(df_ml)), random_state=42)
            fig3d  = px.scatter_3d(sample, x="PCA_1", y="PCA_2", z="PCA_3",
                color="Cognitive_Pattern", color_discrete_map=PAL,
                hover_data=["Student_ID"] + n_cols[:3], opacity=0.78)
            fig3d.update_traces(marker=dict(size=4, line=dict(width=0.3, color="white")))
            scene_bg = "#1a1a2e" if T["DM"] else "#faf9f5"
            grid_cl  = "#3a3a5a" if T["DM"] else "#e8e6df"
            fig3d.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=640,
                margin=dict(l=0,r=0,b=0,t=20),
                scene=dict(bgcolor=scene_bg,
                    xaxis=dict(gridcolor=grid_cl, title=dict(text="PC 1", font=dict(color=CLR_TEXT))),
                    yaxis=dict(gridcolor=grid_cl, title=dict(text="PC 2", font=dict(color=CLR_TEXT))),
                    zaxis=dict(gridcolor=grid_cl, title=dict(text="PC 3", font=dict(color=CLR_TEXT)))),
                legend=dict(font=dict(color=CLR_TEXT),
                            bgcolor="rgba(0,0,0,0.3)" if T["DM"] else "rgba(255,255,255,0.85)",
                            bordercolor=CLR_BORDER, borderwidth=1))
            st.plotly_chart(fig3d, use_container_width=True)

        with t4:
            st.markdown("### Machine Learning Engine")
            cL2, cR2 = st.columns(2)
            with cL2:
                st.markdown("#### 🎯 Classification — Logistic Regression")
                yle = LabelEncoder().fit_transform(df_ml["Cognitive_Pattern"])
                Xs  = StandardScaler().fit_transform(df_ml[n_cols])
                Xtr, Xte, ytr, yte = train_test_split(Xs, yle, test_size=0.2, random_state=42)
                clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
                prd = clf.predict(Xte)
                acc = accuracy_score(yte, prd)
                st.markdown(f"""
                <div class="metric-card indigo" style="text-align:center;padding:18px 24px;margin-bottom:16px;">
                    <div style="color:{CLR_SUB};font-size:11px;font-weight:700;letter-spacing:1px;">CLASSIFICATION ACCURACY</div>
                    <div style="color:#4f46e5;font-size:2.2rem;font-weight:800;font-family:'Literata',serif;">{acc:.1%}</div>
                </div>""", unsafe_allow_html=True)
                cm     = confusion_matrix(yte, prd)
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                   labels=dict(x="Predicted", y="Actual"))
                fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=310, margin=dict(t=10,b=10,l=0,r=0), font=dict(color=CLR_TEXT))
                st.plotly_chart(fig_cm, use_container_width=True)

            with cR2:
                st.markdown("#### 📈 Regression & Feature Importance")
                target   = "Assessment_Score" if "Assessment_Score" in df_ml.columns else n_cols[-1]
                Xreg     = df_ml[[c for c in n_cols if c != target]]
                yreg     = df_ml[target]
                Xrtr, Xrte, yrtr, yrte = train_test_split(Xreg, yreg, test_size=0.2, random_state=42)
                lr = LinearRegression().fit(Xrtr, yrtr)
                r2 = r2_score(yrte, lr.predict(Xrte))
                st.markdown(f"""
                <div class="metric-card green" style="text-align:center;padding:18px 24px;margin-bottom:16px;">
                    <div style="color:{CLR_SUB};font-size:11px;font-weight:700;letter-spacing:1px;">LINEAR REGRESSION R²</div>
                    <div style="color:#16a34a;font-size:2.2rem;font-weight:800;font-family:'Literata',serif;">{r2:.3f}</div>
                </div>""", unsafe_allow_html=True)
                rf  = RandomForestClassifier(n_estimators=80, random_state=42).fit(Xreg, df_ml["Cognitive_Pattern"])
                imp = pd.DataFrame({"Feature": Xreg.columns, "Importance": rf.feature_importances_}).sort_values("Importance")
                fig_imp = go.Figure(go.Bar(
                    y=imp["Feature"], x=imp["Importance"], orientation="h",
                    marker_color="#4f46e5", marker_line_width=0,
                    text=[f"{v:.3f}" for v in imp["Importance"]], textposition="outside",
                    textfont=dict(color=CLR_TEXT, size=11)))
                fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=290, margin=dict(t=20,b=0,l=0,r=60),
                    title=dict(text="Random Forest Feature Importance", font=dict(size=13, color=CLR_TEXT)),
                    xaxis=dict(title="Importance", tickfont=dict(color=CLR_SUB), gridcolor=CLR_SOFT),
                    yaxis=dict(tickfont=dict(color=CLR_TEXT)), font=dict(color=CLR_TEXT))
                st.plotly_chart(fig_imp, use_container_width=True)

            st.markdown("#### 🏆 Multi-Model Classification Comparison")
            models = {
                "Logistic Regression":  LogisticRegression(max_iter=500),
                "Random Forest":        RandomForestClassifier(n_estimators=50, random_state=42),
                "Gradient Boosting":    GradientBoostingClassifier(n_estimators=50, random_state=42),
                "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=7),
                "Naïve Bayes":          GaussianNB(),
            }
            model_results = []
            for mname, model in models.items():
                try:
                    model.fit(Xtr, ytr)
                    model_results.append({"Model": mname, "Accuracy": accuracy_score(yte, model.predict(Xte)) * 100})
                except Exception:
                    pass
            if model_results:
                df_mod  = pd.DataFrame(model_results).sort_values("Accuracy", ascending=False)
                fig_mod = go.Figure(go.Bar(
                    x=df_mod["Model"], y=df_mod["Accuracy"],
                    marker_color=["#4f46e5" if i==0 else "#818cf8" for i in range(len(df_mod))],
                    marker_line_width=0,
                    text=[f"{v:.1f}%" for v in df_mod["Accuracy"]], textposition="outside",
                    textfont=dict(color=CLR_TEXT)))
                fig_mod.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=320, margin=dict(t=20,b=0,l=0,r=0),
                    yaxis=dict(title="Accuracy %", range=[0,110], tickfont=dict(color=CLR_SUB), gridcolor=CLR_SOFT),
                    xaxis=dict(tickfont=dict(color=CLR_TEXT, size=11)),
                    font=dict(color=CLR_TEXT))
                st.plotly_chart(fig_mod, use_container_width=True)
    else:
        st.markdown(f"""
        <div style="text-align:center;padding:80px 40px;color:{CLR_SUB};">
            <div style="font-size:64px;margin-bottom:20px;">📊</div>
            <h2 style="color:{CLR_TEXT};font-family:'Literata',serif;">Upload or Generate a Dataset</h2>
            <p>Upload a CSV or synthesise student data to explore the AI analytics engine.</p>
        </div>""", unsafe_allow_html=True)
