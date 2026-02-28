# =============================================================================
# core/engine.py  –  Text extraction, local NLP engine, Claude AI helpers,
#                    quiz / exam generation (local fallback & AI-powered)
# =============================================================================

import re
import io
import json
import string
import collections

import numpy as np
import requests

from config.data import STOPWORDS, DIFFICULTY_SETTINGS


# =============================================================================
# TEXT EXTRACTION
# =============================================================================

def extract_text(uploaded_file) -> str:
    """Read an uploaded file (PDF / DOCX / PPTX / TXT) and return plain text."""
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
            import zipfile
            import xml.etree.ElementTree as ET
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
                or re.match(r"^(Chapter|Unit|Section|Module|\d+[\.\)])", line, re.I)
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
    """Send a prompt to the Anthropic Messages API.  Returns '' on missing key."""
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
    """Extract the first JSON array or object from *text*."""
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

    # Definition-based MCQs
    for d in defs[:min(4, num_q // 3 + 1)]:
        if qnum > num_q:
            break
        term    = d["term"]
        correct = d["definition"][:100]
        distr   = [
            f"A method for eliminating {term}.",
            f"The inverse of {term}.",
            f"Unrelated to {term}.",
        ][:nd]
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

    # Fill-in-the-blank from sentences
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

    # Multi-select from keywords
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

    # Statement-accuracy MCQs
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
