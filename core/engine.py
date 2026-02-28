# =============================================================================
# core/engine.py  –  Text extraction, local NLP, Claude AI helpers,
#                    quiz/exam generation, AND behavioral analysis engine.
# =============================================================================

import re
import io
import json
import string
import math
import datetime
import collections

import numpy as np
import requests

from config.data import STOPWORDS, DIFFICULTY_SETTINGS


# =============================================================================
# TEXT EXTRACTION
# =============================================================================

def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    raw  = uploaded_file.read()
    text = ""
    try:
        if name.endswith(".pdf"):
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages[:25])
        elif name.endswith(".docx"):
            import docx as _docx
            doc  = _docx.Document(io.BytesIO(raw))
            text = "\n".join(p.text for p in doc.paragraphs)
        elif name.endswith(".pptx"):
            import zipfile, xml.etree.ElementTree as ET
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                slides = [f for f in z.namelist()
                          if f.startswith("ppt/slides/slide") and f.endswith(".xml")]
                for s in slides[:20]:
                    tree = ET.fromstring(z.read(s))
                    text += " ".join(t.text for t in tree.iter() if t.text) + "\n"
        else:
            text = raw.decode("utf-8", errors="ignore")
    except Exception as e:
        text = f"[Error reading file: {e}]"
    text = re.sub(r"[ \t]{3,}", " ", text)
    text = re.sub(r"\n{4,}", "\n\n", text)
    return text.strip()


# =============================================================================
# LOCAL NLP UTILITIES
# =============================================================================

def tokenize(text: str) -> list:
    return re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

def extract_keywords(text: str, top_n: int = 30) -> list:
    words = [w for w in tokenize(text) if w not in STOPWORDS]
    freq  = collections.Counter(words)
    return [w for w, _ in freq.most_common(top_n)]

def split_sentences(text: str) -> list:
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if 30 < len(s.strip()) < 500]

def score_sentence(sent: str, keywords: list) -> int:
    words = set(tokenize(sent))
    return sum(1 for kw in keywords if kw in words)

def extract_definitions(text: str) -> list:
    patterns = [
        r"([A-Z][a-zA-Z\s]{2,30})\s+(?:is defined as|is|refers to|means|denotes)\s+([^.]{15,150}\.)",
        r"([A-Z][a-zA-Z\s]{2,25})\s*:\s*([^.\n]{15,150})",
        r"(?:Definition|Define|Term)[:—]\s*([A-Z][a-zA-Z\s]{2,25})[—:]\s*([^.\n]{15,120})",
    ]
    defs, seen = [], set()
    for pat in patterns:
        for m in re.finditer(pat, text):
            term = m.group(1).strip()
            defn = m.group(2).strip()
            if term.lower() not in seen and len(term) > 3 and len(defs) < 12:
                seen.add(term.lower())
                defs.append({"term": term, "definition": defn})
    return defs

def extract_headings(text: str) -> list:
    lines    = text.split("\n")
    headings = []
    for line in lines:
        line = line.strip()
        if (
            10 < len(line) < 80
            and (
                line.isupper()
                or re.match(r"^(Chapter|Unit|Section|Module|\d+[\.)])", line, re.I)
                or (line[0].isupper() and not line.endswith(".") and len(line.split()) <= 8)
            )
        ):
            headings.append(line)
    return list(dict.fromkeys(headings))[:12]

def local_summarize(text: str, n: int = 7) -> list:
    sents    = split_sentences(text)
    if not sents:
        return ["No readable content found in document."]
    keywords = extract_keywords(text, 35)
    scored   = sorted(enumerate(sents), key=lambda x: score_sentence(x[1], keywords), reverse=True)
    top      = sorted(scored[:n], key=lambda x: x[0])
    return [s for _, s in top]

def local_generate_summary(texts: list) -> dict:
    combined  = "\n\n".join(texts)
    keywords  = extract_keywords(combined, 15)
    key_sents = local_summarize(combined, 7)
    headings  = extract_headings(combined)
    defs      = extract_definitions(combined)
    topics    = headings[:6] if headings else [kw.title() for kw in keywords[:6]]
    return {
        "key_topics":           topics,
        "core_definitions":     defs if defs else [
            {"term": kw.title(), "definition": f"Key concept: {kw}."} for kw in keywords[:4]
        ],
        "important_principles": key_sents[:3],
        "exam_topics":          [kw.title() for kw in keywords[:8]],
        "quick_review":         key_sents[3:7],
        "document_overview":    key_sents[0] if key_sents else "Document processed.",
        "keywords":             keywords,
    }


# =============================================================================
# CLAUDE API HELPER
# =============================================================================

def call_claude(prompt: str, api_key: str, max_tokens: int = 3000) -> str:
    if not api_key:
        return ""
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":          api_key,
                "anthropic-version":  "2023-06-01",
                "Content-Type":       "application/json",
            },
            json={
                "model":      "claude-opus-4-5",
                "max_tokens": max_tokens,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()["content"][0]["text"]
        return f"API_ERROR:{r.status_code}:{r.text[:200]}"
    except Exception as e:
        return f"CONN_ERROR:{e}"


def extract_json(text: str, kind: str = "array"):
    bracket = ("[", "]") if kind == "array" else ("{", "}")
    m = re.search(re.escape(bracket[0]) + r".*?" + re.escape(bracket[1]), text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


# =============================================================================
# AI-POWERED GENERATION (Claude)
# =============================================================================

def ai_summarize(file_texts: list, subject: str, api_key: str) -> dict | None:
    combined = "\n\n---\n\n".join(f"[Document {i+1}]\n{t}" for i, t in enumerate(file_texts))
    prompt = f"""You are an expert academic tutor. The student has uploaded study material for: **{subject}**.

FULL TEXT:
{combined[:5000]}

Return ONLY valid JSON:
{{
  "key_topics": ["topic 1","topic 2","topic 3","topic 4","topic 5"],
  "core_definitions": [
    {{"term": "term1", "definition": "definition1"}},
    {{"term": "term2", "definition": "definition2"}}
  ],
  "important_principles": ["principle 1","principle 2","principle 3"],
  "exam_topics": ["high-priority topic 1","topic 2","topic 3","topic 4"],
  "quick_review": ["bullet 1","bullet 2","bullet 3","bullet 4","bullet 5"],
  "document_overview": "2-3 sentence summary"
}}"""
    raw = call_claude(prompt, api_key, 2000)
    if not raw or raw.startswith("API_ERROR") or raw.startswith("CONN_ERROR"):
        return None
    return extract_json(raw, "object")


def ai_generate_quiz(file_texts: list, subject: str, num_q: int,
                     difficulty: str, api_key: str) -> list | None:
    combined = "\n\n".join(f"[Doc {i+1}]\n{t}" for i, t in enumerate(file_texts))
    prompt = f"""You are a university exam setter for {subject}. Difficulty: {difficulty}.

DOCUMENT CONTENT:
{combined[:4500]}

Generate exactly {num_q} exam questions (MCQ, MSQ, BLANK types, distributed evenly).
Return ONLY valid JSON array:
[
  {{"num":1,"type":"MCQ","question":"...","options":["Correct","Wrong B","Wrong C","Wrong D"],"answer":"A","explanation":"...","topic":"...", "difficulty":"{difficulty}"}},
  {{"num":2,"type":"MSQ","question":"...","options":["Correct A","Wrong B","Correct C","Wrong D"],"answer":["A","C"],"explanation":"...","topic":"...","difficulty":"{difficulty}"}},
  {{"num":3,"type":"BLANK","question":"The __________ is...","answer":"exact term","explanation":"...","topic":"...","difficulty":"{difficulty}"}}
]
Generate all {num_q} questions specific to the uploaded content."""
    raw = call_claude(prompt, api_key, 4000)
    if not raw or raw.startswith("API_ERROR") or raw.startswith("CONN_ERROR"):
        return None
    return extract_json(raw, "array")


def ai_generate_exam(file_texts: list, subject: str, n2m: int, n16m: int,
                     api_key: str) -> dict | None:
    combined = "\n\n".join(f"[Doc {i+1}]\n{t}" for i, t in enumerate(file_texts))
    prompt = f"""You are a university exam paper setter for {subject}.

DOCUMENT CONTENT:
{combined[:4500]}

Generate a formal university exam:
- Part A: {n2m} short answer questions (2 marks each)
- Part B: {n16m} detailed essay questions (16 marks each)

Return ONLY valid JSON:
{{
  "subject": "{subject}",
  "part_a": [{{"num":1,"question":"Define [term].","marks":2}}],
  "part_b": [{{"num":{n2m+1},"question":"Explain in detail [concept] with examples.","marks":16}}]
}}
Generate ALL {n2m} Part A and {n16m} Part B questions."""
    raw = call_claude(prompt, api_key, 3000)
    if not raw or raw.startswith("API_ERROR") or raw.startswith("CONN_ERROR"):
        return None
    return extract_json(raw, "object")


def ai_recommendations(responses: dict, course_prog: int, subject: str,
                        api_key: str) -> list | None:
    total   = len(responses)
    correct = sum(1 for r in responses.values() if r.get("correct", False))
    score   = (correct / total * 100) if total else 0
    weak    = [r.get("topic", "")[:60] for r in responses.values()
               if not r.get("correct", True)]
    prompt = f"""You are an AI academic coach.
- Subject: {subject}
- Quiz score: {score:.0f}% ({correct}/{total} correct)
- Course completion: {course_prog}%
- Weak areas: {'; '.join(weak[:4]) if weak else 'None'}

Return ONLY valid JSON array of 4 recommendations:
[{{"priority":"high","title":"Short title","description":"2 sentences.","action":"Concrete next step","time_est":"e.g. 2 hours","impact":"Expected outcome"}}]
Use priorities: high/medium/low."""
    raw = call_claude(prompt, api_key, 1200)
    if not raw or raw.startswith("API_ERROR") or raw.startswith("CONN_ERROR"):
        return None
    return extract_json(raw, "array")


# =============================================================================
# LOCAL QUIZ & EXAM GENERATION (fallback, no API key required)
# =============================================================================

def local_generate_quiz(texts: list, num_q: int = 10, difficulty: str = "Medium") -> list:
    combined  = "\n\n".join(texts)
    sentences = split_sentences(combined)
    keywords  = extract_keywords(combined, 50)
    defs      = extract_definitions(combined)
    headings  = extract_headings(combined)
    questions = []
    qnum      = 1
    cfg       = DIFFICULTY_SETTINGS[difficulty]
    nd        = cfg["distractors"]

    topic_map = {}
    for i, kw in enumerate(keywords[:20]):
        for h in headings:
            if kw.lower() in h.lower():
                topic_map[kw] = h
                break
        if kw not in topic_map:
            topic_map[kw] = headings[i % len(headings)] if headings else "General"

    for d in defs[:min(4, num_q // 3 + 1)]:
        if qnum > num_q:
            break
        term    = d["term"]
        correct = d["definition"][:100]
        distr   = [f"A method for eliminating {term}.", f"The inverse of {term}.", f"Unrelated to {term}."][:nd]
        opts = [correct] + distr
        np.random.shuffle(opts)
        ai   = opts.index(correct)
        t_kw = tokenize(term)[0] if tokenize(term) else term
        topic = topic_map.get(t_kw, "General")
        questions.append({
            "num": qnum, "type": "MCQ", "difficulty": difficulty, "topic": topic,
            "question": f'Which best defines "{term}"?', "options": opts,
            "answer": chr(65 + ai),
            "explanation": f'"{term}" is defined as: {correct}',
        })
        qnum += 1

    for sent in sentences:
        if qnum > num_q:
            break
        words = [w for w in sent.split()
                 if len(w) > 5 and w[0].isupper() and w.lower() not in STOPWORDS]
        if not words:
            continue
        target  = words[0]
        blank_q = sent.replace(target, "________", 1)
        if blank_q == sent:
            continue
        kw_match = next((kw for kw in keywords if kw in target.lower()), keywords[0] if keywords else "General")
        questions.append({
            "num": qnum, "type": "BLANK", "difficulty": difficulty,
            "topic": topic_map.get(kw_match, "General"),
            "question": blank_q,
            "answer": target.strip(string.punctuation),
            "explanation": f'The answer is "{target.strip(string.punctuation)}".',
        })
        qnum += 1

    if qnum <= num_q and len(keywords) >= 8:
        correct_kws = keywords[:3]
        wrong_kws   = keywords[5:8]
        opts        = [kw.title() for kw in correct_kws + wrong_kws]
        np.random.shuffle(opts)
        correct_letters = sorted([chr(65 + opts.index(kw.title())) for kw in correct_kws])
        questions.append({
            "num": qnum, "type": "MSQ", "difficulty": difficulty,
            "topic": headings[0] if headings else "General",
            "question": "Which are key concepts in this document? (Select ALL that apply)",
            "options": opts, "answer": correct_letters,
            "explanation": f'Key concepts: {", ".join(kw.title() for kw in correct_kws)}',
        })
        qnum += 1

    for sent in sentences[5:]:
        if qnum > num_q:
            break
        if len(sent.split()) < 8:
            continue
        kw_match = next((kw for kw in keywords if kw in sent.lower()), "General")
        questions.append({
            "num": qnum, "type": "MCQ", "difficulty": difficulty,
            "topic": topic_map.get(kw_match, "General"),
            "question": f'Based on the document:\n"{sent[:130]}..."',
            "options": [
                "This statement is accurate as per the document.",
                "This statement is completely false.",
                "This applies only to unrelated domains.",
                "The document does not address this topic.",
            ][:nd + 1],
            "answer": "A",
            "explanation": "Directly extracted from the uploaded document.",
        })
        qnum += 1

    return questions[:num_q]


def local_generate_exam(texts: list, subject: str, n2m: int = 10, n16m: int = 3) -> dict:
    combined = "\n\n".join(texts)
    keywords = extract_keywords(combined, 30)
    sents    = split_sentences(combined)
    defs     = extract_definitions(combined)
    headings = extract_headings(combined)

    part_a = []
    for i in range(n2m):
        if i < len(defs):
            q = f'Define "{defs[i]["term"]}" in the context of {subject}.'
        elif i < len(keywords):
            q = f'State the significance of "{keywords[i].title()}" in {subject}.'
        elif i < len(sents):
            q = f'Briefly explain: "{sents[i][:100]}..."'
        else:
            q = f"List two key points related to {subject}."
        part_a.append({"num": i + 1, "question": q, "marks": 2})

    pool   = headings if headings else [kw.title() for kw in keywords[:n16m * 2]]
    part_b = []
    for j in range(n16m):
        topic = pool[j] if j < len(pool) else subject
        q     = (
            f'Discuss in detail "{topic}" as covered in the study material. '
            f"Include definitions, principles, and practical applications with examples."
        )
        part_b.append({"num": n2m + j + 1, "question": q, "marks": 16})

    return {"subject": subject, "part_a": part_a, "part_b": part_b}


# =============================================================================
# ███████████████████████████████████████████████████████████████████████████
# BEHAVIORAL ANALYSIS ENGINE  (Problem-statement core requirement)
# ███████████████████████████████████████████████████████████████████████████
# =============================================================================

# ── Cognitive Pattern Labels ──────────────────────────────────────────────────
COGNITIVE_PATTERNS = {
    "Rapid Master":        {"desc": "Fast, accurate responses. High confidence, low retry rate.", "color": "#22c55e", "icon": "🚀"},
    "Analytical Thinker":  {"desc": "Deliberate, thoughtful. Slower but highly accurate.", "color": "#3b82f6", "icon": "🔬"},
    "Impulsive Responder": {"desc": "Quick but error-prone. Rushes without verifying answers.", "color": "#f59e0b", "icon": "⚡"},
    "Foundational Learner":{"desc": "Struggles with retention. Needs structured review and repetition.", "color": "#ef4444", "icon": "📚"},
    "Strategic Revisor":   {"desc": "Uses retries effectively. Learns actively from mistakes.", "color": "#8b5cf6", "icon": "🔄"},
}

# ── Behavioral Feature Extraction ─────────────────────────────────────────────

def compute_behavioral_features(quiz_history: list, retry_map: dict) -> dict:
    """
    Extract behavioral metrics from quiz_history:
      - avg_response_time_s  : mean seconds per question
      - mistake_frequency    : proportion of wrong answers overall
      - retry_rate           : avg retry count per quiz subject
      - speed_accuracy_ratio : speed (1/time) vs accuracy
      - topic_consistency    : std of per-topic accuracy (high = inconsistent)
      - improvement_slope    : regression slope of scores over time
      - difficulty_bias      : preference / avoidance of hard questions
    """
    if not quiz_history:
        return {}

    all_dur        = [q.get("duration_s", 0) for q in quiz_history]
    all_totals     = [q.get("total", 1)      for q in quiz_history]
    all_correct    = [q.get("correct", 0)    for q in quiz_history]
    all_wrong      = [q.get("wrong", 0)      for q in quiz_history]
    all_scores     = [q.get("score", 0)      for q in quiz_history]

    # Per-question response time (seconds)
    per_q_times = [
        d / max(t, 1) for d, t in zip(all_dur, all_totals) if d > 0
    ]
    avg_resp_time = float(np.mean(per_q_times)) if per_q_times else 60.0

    # Mistake frequency (0-1)
    total_qs    = sum(all_totals)
    total_wrong = sum(all_wrong)
    mistake_freq = total_wrong / max(total_qs, 1)

    # Retry rate
    retry_vals  = list(retry_map.values())
    avg_retry   = float(np.mean(retry_vals)) if retry_vals else 1.0

    # Speed-accuracy ratio: accuracy divided by normalized time (lower time = faster)
    acc_overall = 1 - mistake_freq
    norm_speed  = 1 / (1 + avg_resp_time / 60)  # 0-1, higher = faster
    sar         = acc_overall * norm_speed

    # Topic consistency: std of per-topic accuracy across all quizzes
    topic_acc_map: dict = collections.defaultdict(lambda: {"c": 0, "t": 0})
    for q in quiz_history:
        for r in q.get("responses", {}).values():
            tp = r.get("topic", "General")
            topic_acc_map[tp]["t"] += 1
            if r.get("correct"):
                topic_acc_map[tp]["c"] += 1
    topic_pcts = [
        v["c"] / v["t"] for v in topic_acc_map.values() if v["t"] > 0
    ]
    topic_consistency = 1 - float(np.std(topic_pcts)) if topic_pcts else 0.5

    # Improvement slope (linear regression on scores)
    if len(all_scores) >= 2:
        x = np.arange(len(all_scores))
        slope = float(np.polyfit(x, all_scores, 1)[0])
    else:
        slope = 0.0

    # Difficulty bias score (Hard % of attempts)
    diff_counts = collections.Counter(q.get("difficulty", "Medium") for q in quiz_history)
    total_diff  = sum(diff_counts.values())
    hard_bias   = diff_counts.get("Hard", 0) / max(total_diff, 1)

    return {
        "avg_response_time_s":  round(avg_resp_time, 1),
        "mistake_frequency":    round(mistake_freq, 3),
        "avg_retry_rate":       round(avg_retry, 2),
        "speed_accuracy_ratio": round(sar, 3),
        "topic_consistency":    round(topic_consistency, 3),
        "improvement_slope":    round(slope, 3),
        "hard_difficulty_bias": round(hard_bias, 3),
        "total_questions":      total_qs,
        "total_attempts":       len(quiz_history),
        "avg_score":            round(float(np.mean(all_scores)) if all_scores else 0, 1),
    }


def classify_cognitive_pattern(features: dict) -> str:
    """
    Rule-based pattern classifier using behavioral features.
    Returns one of the COGNITIVE_PATTERNS keys.
    """
    if not features:
        return "Foundational Learner"

    acc   = 1 - features.get("mistake_frequency", 0.5)
    speed = 1 / (1 + features.get("avg_response_time_s", 60) / 60)
    retry = features.get("avg_retry_rate", 1.0)
    slope = features.get("improvement_slope", 0)
    cons  = features.get("topic_consistency", 0.5)
    avg_s = features.get("avg_score", 50)

    # Strategic Revisor: retries a lot AND improves
    if retry >= 2.5 and slope >= 2.0:
        return "Strategic Revisor"

    # Rapid Master: fast AND accurate AND consistent
    if speed >= 0.55 and acc >= 0.75 and cons >= 0.65:
        return "Rapid Master"

    # Impulsive Responder: fast but inaccurate AND inconsistent
    if speed >= 0.50 and acc < 0.65 and cons < 0.55:
        return "Impulsive Responder"

    # Analytical Thinker: slow but accurate
    if speed < 0.45 and acc >= 0.72:
        return "Analytical Thinker"

    # Foundational Learner: low accuracy, no improvement
    if avg_s < 55 or (acc < 0.55 and slope <= 1.0):
        return "Foundational Learner"

    # Default: nearest to Analytical
    return "Analytical Thinker"


def compute_per_quiz_behavior(quiz_entry: dict) -> dict:
    """Extract per-quiz behavioral signals for trend charts."""
    responses = quiz_entry.get("responses", {})
    total     = max(len(responses), 1)
    dur       = quiz_entry.get("duration_s", 0)
    correct   = quiz_entry.get("correct", 0)
    wrong     = quiz_entry.get("wrong", total - correct)

    # Response time per question
    time_per_q = dur / total if dur else 0

    # Question-level difficulty distribution
    diffs   = [r.get("difficulty", "Medium") for r in responses.values()]
    hard_n  = diffs.count("Hard")
    easy_n  = diffs.count("Easy")

    # First-attempt accuracy (all answers given before retry)
    accuracy = correct / total

    # Topic-level accuracy for weak-spot detection
    topic_acc: dict = collections.defaultdict(lambda: {"c": 0, "t": 0})
    for r in responses.values():
        tp = r.get("topic", "General")
        topic_acc[tp]["t"] += 1
        if r.get("correct"):
            topic_acc[tp]["c"] += 1

    weak_topics = [
        tp for tp, v in topic_acc.items()
        if v["t"] > 0 and (v["c"] / v["t"]) < 0.5
    ]
    strong_topics = [
        tp for tp, v in topic_acc.items()
        if v["t"] > 0 and (v["c"] / v["t"]) >= 0.8
    ]

    return {
        "time_per_q_s":   round(time_per_q, 1),
        "accuracy":       round(accuracy, 3),
        "wrong_count":    wrong,
        "hard_questions": hard_n,
        "easy_questions": easy_n,
        "weak_topics":    weak_topics,
        "strong_topics":  strong_topics,
        "score":          quiz_entry.get("score", 0),
        "difficulty":     quiz_entry.get("difficulty", "Medium"),
        "subject":        quiz_entry.get("subject", ""),
        "ts":             quiz_entry.get("ts", ""),
    }


# ── Adaptive Recommendation Engine ────────────────────────────────────────────

STRATEGY_MAP = {
    "Rapid Master": {
        "study_style":    "Challenge-Based Learning",
        "session_length": "45-60 min deep sessions",
        "review_freq":    "Weekly cumulative reviews",
        "focus":          "Hard problems and edge cases",
    },
    "Analytical Thinker": {
        "study_style":    "Structured Problem Solving",
        "session_length": "60-90 min focused sessions",
        "review_freq":    "Bi-weekly with self-testing",
        "focus":          "Speed-building timed exercises",
    },
    "Impulsive Responder": {
        "study_style":    "Slow-Down Technique",
        "session_length": "30 min with mandatory review",
        "review_freq":    "Daily short quizzes",
        "focus":          "Reading questions fully before answering",
    },
    "Foundational Learner": {
        "study_style":    "Spaced Repetition",
        "session_length": "20-30 min daily",
        "review_freq":    "Daily — use flashcards",
        "focus":          "Core concepts and definitions",
    },
    "Strategic Revisor": {
        "study_style":    "Active Recall + Elaboration",
        "session_length": "45 min with 10-min review",
        "review_freq":    "Every 3 days",
        "focus":          "Consolidating learned material faster",
    },
}


def generate_adaptive_recommendations(
    features: dict,
    pattern: str,
    quiz_history: list,
    retry_map: dict,
    overall_progress: int,
) -> list:
    """
    Generate data-driven adaptive recommendations entirely locally
    without requiring an API key.
    """
    if not features:
        return [{
            "priority": "high",
            "title": "Start Your First Quiz",
            "description": "Complete at least one quiz so the system can analyse your learning patterns.",
            "action": "Go to Brain Training → Upload materials → Generate Quiz",
            "time_est": "20 min",
            "impact": "Unlocks personalised adaptive recommendations",
            "evidence": "No behavioral data yet",
        }]

    recs      = []
    strategy  = STRATEGY_MAP.get(pattern, STRATEGY_MAP["Foundational Learner"])
    acc       = 1 - features["mistake_frequency"]
    avg_time  = features["avg_response_time_s"]
    avg_retry = features["avg_retry_rate"]
    slope     = features["improvement_slope"]
    avg_score = features["avg_score"]
    cons      = features["topic_consistency"]

    # Recommendation 1: Pattern-based study strategy
    recs.append({
        "priority": "high",
        "title":    f"Adopt {strategy['study_style']}",
        "description": (
            f"As a '{pattern}', you benefit most from {strategy['study_style'].lower()}. "
            f"Your data shows {acc*100:.0f}% accuracy with {avg_time:.0f}s avg per question."
        ),
        "action":   f"Use {strategy['session_length']} — {strategy['focus']}",
        "time_est": strategy["session_length"],
        "impact":   f"Expected +10-15% accuracy within 2 weeks",
        "evidence": f"Pattern: {pattern} | Score: {avg_score:.0f}% | Accuracy: {acc*100:.0f}%",
    })

    # Recommendation 2: Response time coaching
    if avg_time > 90:
        recs.append({
            "priority": "medium",
            "title":    "Build Response Speed",
            "description": (
                f"You average {avg_time:.0f}s per question — above optimal (60s). "
                "This may indicate uncertainty in foundational concepts."
            ),
            "action":   "Practice 5-minute timed mini-quizzes daily to build fluency",
            "time_est": "5-10 min/day",
            "impact":   "Reduce avg time to <60s within 1 week",
            "evidence": f"Avg response time: {avg_time:.0f}s (target: ≤60s)",
        })
    elif avg_time < 25 and acc < 0.70:
        recs.append({
            "priority": "high",
            "title":    "Slow Down — Read Carefully",
            "description": (
                f"You answer in {avg_time:.0f}s but accuracy is {acc*100:.0f}%. "
                "Speed without accuracy costs marks. Re-read each question fully."
            ),
            "action":   "Set minimum 30s per question rule. Review all answers before submitting.",
            "time_est": "No extra time — just pace yourself",
            "impact":   "Expected +20% accuracy immediately",
            "evidence": f"Avg time: {avg_time:.0f}s | Accuracy: {acc*100:.0f}%",
        })

    # Recommendation 3: Retry pattern analysis
    if avg_retry >= 3.0:
        recs.append({
            "priority": "medium",
            "title":    "Leverage Your Retry Habit",
            "description": (
                f"You retry quizzes {avg_retry:.1f}× on average — a strong learning signal. "
                "You learn actively from mistakes. Now focus on reducing first-attempt errors."
            ),
            "action":   f"Before retrying, write down why each wrong answer failed. Target: <2 retries per subject.",
            "time_est": "5 min reflection per quiz",
            "impact":   "Convert retries into first-attempt success",
            "evidence": f"Avg retry rate: {avg_retry:.1f}×",
        })

    # Recommendation 4: Improvement trend
    if slope > 3.0:
        recs.append({
            "priority": "low",
            "title":    "Sustain Your Momentum",
            "description": (
                f"Your scores are improving at +{slope:.1f}% per attempt — excellent trajectory. "
                "Keep the current rhythm to maintain growth."
            ),
            "action":   strategy["review_freq"],
            "time_est": "Current pace",
            "impact":   "Reach 90%+ average within next 5 quizzes",
            "evidence": f"Score slope: +{slope:.1f}%/attempt",
        })
    elif slope < -2.0:
        recs.append({
            "priority": "high",
            "title":    "Reverse Score Decline",
            "description": (
                f"Your scores are declining by {abs(slope):.1f}% per attempt — a warning sign of fatigue or gaps. "
                "Reduce frequency and deepen review."
            ),
            "action":   "Take a 2-day break, then restart with Easy difficulty quizzes on weakest topics.",
            "time_est": "2 days rest + 30 min review",
            "impact":   "Reset learning curve and rebuild confidence",
            "evidence": f"Score slope: {slope:.1f}%/attempt",
        })

    # Recommendation 5: Topic consistency
    if cons < 0.40:
        # Find weakest topics from recent quizzes
        all_weak = []
        for q in quiz_history[-3:]:
            bhv = compute_per_quiz_behavior(q)
            all_weak.extend(bhv.get("weak_topics", []))
        weak_uniq = list(dict.fromkeys(all_weak))[:3]
        recs.append({
            "priority": "high",
            "title":    "Fix Topic Inconsistency",
            "description": (
                f"Your accuracy varies widely across topics (consistency score: {cons*100:.0f}%). "
                f"Weakest areas: {', '.join(weak_uniq) if weak_uniq else 'multiple topics'}."
            ),
            "action":   f"Dedicate 20 min each to: {', '.join(weak_uniq[:2]) if weak_uniq else 'your identified weak topics'}",
            "time_est": "20 min × 3 sessions",
            "impact":   "Raise topic consistency to 70%+",
            "evidence": f"Topic consistency: {cons*100:.0f}% (target: 70%+)",
        })

    # Recommendation 6: Overall progress
    if overall_progress < 30:
        recs.append({
            "priority": "medium",
            "title":    "Accelerate Course Completion",
            "description": (
                f"Course completion is {overall_progress}%. Pairing active quizzing "
                "with lesson completion accelerates long-term retention."
            ),
            "action":   "Complete 1 lesson + 1 quiz per study session",
            "time_est": "45 min/session",
            "impact":   "Reach 50% course completion this week",
            "evidence": f"Current progress: {overall_progress}%",
        })

    return recs[:6]


# ── Improvement Analytics Report ──────────────────────────────────────────────

def generate_improvement_report(
    quiz_history: list,
    retry_map: dict,
    features: dict,
    pattern: str,
    overall_progress: int,
) -> dict:
    """
    Produce a structured improvement analytics report with:
    - Behavioral summary
    - Strengths and weaknesses
    - Milestone tracking
    - Week-over-week score change
    - Predicted next score
    """
    if not quiz_history:
        return {"error": "No quiz history available."}

    all_scores = [q.get("score", 0) for q in quiz_history]
    avg_score  = float(np.mean(all_scores)) if all_scores else 0
    best_score = max(all_scores) if all_scores else 0
    worst_score= min(all_scores) if all_scores else 0

    # Week-over-week change
    wow_change = 0.0
    if len(all_scores) >= 2:
        mid   = len(all_scores) // 2
        first = np.mean(all_scores[:mid])
        last  = np.mean(all_scores[mid:])
        wow_change = float(last - first)

    # Predicted next score (linear extrapolation)
    predicted_next = avg_score
    if len(all_scores) >= 2:
        slope = features.get("improvement_slope", 0)
        predicted_next = min(100, max(0, all_scores[-1] + slope))

    # Milestone achievements
    milestones = []
    if best_score >= 90:
        milestones.append({"label": "Excellence", "desc": "Scored 90%+ in a quiz", "icon": "🏆"})
    if best_score >= 70:
        milestones.append({"label": "Proficiency", "desc": "Passed at least one quiz", "icon": "✅"})
    if len(quiz_history) >= 5:
        milestones.append({"label": "Committed Learner", "desc": "Completed 5+ quizzes", "icon": "🎯"})
    if len(quiz_history) >= 10:
        milestones.append({"label": "Dedicated Scholar", "desc": "Completed 10+ quizzes", "icon": "📚"})
    if overall_progress >= 50:
        milestones.append({"label": "Halfway Hero", "desc": "50%+ course completion", "icon": "⚡"})
    if features.get("improvement_slope", 0) > 2.0:
        milestones.append({"label": "Rising Star", "desc": "Consistent score improvement", "icon": "🌟"})

    # All-time weak and strong topics
    topic_agg: dict = collections.defaultdict(lambda: {"c": 0, "t": 0})
    for q in quiz_history:
        for r in q.get("responses", {}).values():
            tp = r.get("topic", "General")
            topic_agg[tp]["t"] += 1
            if r.get("correct"):
                topic_agg[tp]["c"] += 1

    topic_pcts = {tp: v["c"] / v["t"] for tp, v in topic_agg.items() if v["t"] >= 2}
    strong_topics = sorted(topic_pcts, key=topic_pcts.get, reverse=True)[:3]
    weak_topics   = sorted(topic_pcts, key=topic_pcts.get)[:3]

    # Behavioral strengths and weaknesses
    strengths, weaknesses = [], []

    acc = 1 - features.get("mistake_frequency", 0.5)
    if acc >= 0.75:
        strengths.append(f"High accuracy ({acc*100:.0f}%) — you answer correctly most of the time")
    else:
        weaknesses.append(f"Low accuracy ({acc*100:.0f}%) — review core concepts")

    time_s = features.get("avg_response_time_s", 60)
    if time_s <= 60 and acc >= 0.70:
        strengths.append(f"Efficient response time ({time_s:.0f}s/q) with good accuracy")
    elif time_s > 90:
        weaknesses.append(f"Slow response time ({time_s:.0f}s/q) — practice under timed conditions")

    slope = features.get("improvement_slope", 0)
    if slope >= 2.0:
        strengths.append(f"Improving score trajectory (+{slope:.1f}%/attempt)")
    elif slope < -1.0:
        weaknesses.append(f"Declining scores ({slope:.1f}%/attempt) — needs intervention")

    cons = features.get("topic_consistency", 0.5)
    if cons >= 0.65:
        strengths.append("Consistent performance across multiple topics")
    else:
        weaknesses.append("Inconsistent across topics — some areas much weaker than others")

    return {
        "avg_score":       round(avg_score, 1),
        "best_score":      round(best_score, 1),
        "worst_score":     round(worst_score, 1),
        "wow_change":      round(wow_change, 1),
        "predicted_next":  round(predicted_next, 1),
        "total_quizzes":   len(quiz_history),
        "milestones":      milestones,
        "strong_topics":   strong_topics,
        "weak_topics":     weak_topics,
        "strengths":       strengths,
        "weaknesses":      weaknesses,
        "pattern":         pattern,
        "pattern_info":    COGNITIVE_PATTERNS.get(pattern, {}),
        "features":        features,
    }
