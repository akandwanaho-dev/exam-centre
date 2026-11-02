import os
import json
import re
import hashlib
from datetime import datetime, timezone
from typing import Any

import streamlit as st
import pandas as pd
from pypdf import PdfReader

# ============================================================
# EXAMPREP AI — single-file Streamlit MVP
# Streamlit Community Cloud ready
# ============================================================

APP_NAME = "ExamPrep AI"
SUBJECTS = [
    "Mathematics", "English Language", "Biology", "Chemistry",
    "Physics", "Geography", "History", "Entrepreneurship"
]
LEVELS = ["S1", "S2", "S3", "S4"]

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------- Styling ----------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f6f8fc; }
[data-testid="stSidebar"] { background: #101b33; }
[data-testid="stSidebar"] * { color: #eef3ff !important; }
[data-testid="stSidebar"] .stRadio label { padding: 7px 10px; border-radius: 9px; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #c9d5ee; }
.main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1450px; }
.hero {
  background: linear-gradient(120deg,#14264a 0%,#244c83 65%,#3475b8 100%);
  padding: 28px 30px; border-radius: 20px; color: white; margin-bottom: 20px;
}
.hero h1 { color: white; font-size: 2.1rem; margin: 0 0 7px 0; }
.hero p { color: #dce8fb; margin: 0; font-size: 1rem; }
.metric-card {
  background: white; border: 1px solid #e7ecf4; border-radius: 15px;
  padding: 17px 18px; min-height: 112px;
}
.metric-label { color: #65738b; font-size: .82rem; font-weight: 600; }
.metric-value { color: #172b4d; font-size: 1.8rem; font-weight: 800; margin-top: 5px; }
.metric-note { color: #8290a5; font-size: .76rem; margin-top: 2px; }
.section-title { color: #172b4d; font-weight: 800; font-size: 1.25rem; margin: 8px 0 12px; }
.info-card {
  background: white; border: 1px solid #e7ecf4; border-radius: 14px;
  padding: 17px; height: 100%;
}
.small-muted { color: #718096; font-size: .85rem; }
div.stButton > button { border-radius: 10px; font-weight: 650; min-height: 42px; }
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
  border-radius: 10px;
}
div[data-testid="stChatMessage"] { border-radius: 12px; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------- Session state ----------------------------

DEFAULTS = {
    "role": "Student",
    "full_name": "Student",
    "school_name": "My School",
    "class_level": "S4",
    "subject": "Biology",
    "documents": [],
    "chat_history": [],
    "past_papers": [],
    "generated_exam": None,
    "exam_answers": {},
    "exam_submitted": False,
    "exam_result": None,
    "results": [],
    "revision_notes": [],
    "page": "Home",
    "selected_doc": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------------------- Secrets / AI ----------------------------

def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return os.getenv(name, default)

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-2.5-flash")

@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key: str):
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception:
        return None

def ask_gemini(prompt: str, system_instruction: str = "") -> str:
    client = get_gemini_client(GEMINI_API_KEY)
    if client is None:
        raise RuntimeError(
            "Gemini is not configured. Add GEMINI_API_KEY to Streamlit secrets."
        )
    full_prompt = (
        system_instruction.strip() + "\n\n" + prompt
        if system_instruction else prompt
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=full_prompt,
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("The AI provider returned an empty response. Please retry.")
    return text.strip()

# ---------------------------- Document / RAG helpers ----------------------------

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_into_chunks(text: str, chunk_words: int = 450, overlap: int = 70):
    words = clean_text(text).split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_words - overlap)
    for start in range(0, len(words), step):
        part = words[start:start + chunk_words]
        if not part:
            continue
        chunks.append(" ".join(part))
        if start + chunk_words >= len(words):
            break
    return chunks

def extract_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            content = page.extract_text() or ""
        except Exception:
            content = ""
        if content.strip():
            pages.append({
                "page": page_number,
                "text": clean_text(content)
            })
    return pages, len(reader.pages)

def add_document(uploaded_file, subject, level, doc_type, title):
    raw = uploaded_file.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    if any(d["sha256"] == digest for d in st.session_state.documents):
        return False, "This file has already been added in this session."
    pages, total_pages = extract_pdf(uploaded_file)
    chunks = []
    for page in pages:
        for index, chunk in enumerate(split_into_chunks(page["text"])):
            chunks.append({
                "text": chunk,
                "page": page["page"],
                "chunk_index": index,
                "subject": subject,
                "level": level,
                "doc_type": doc_type,
                "title": title,
                "sha256": digest,
            })
    record = {
        "title": title,
        "file_name": uploaded_file.name,
        "subject": subject,
        "level": level,
        "doc_type": doc_type,
        "pages": total_pages,
        "chunks": chunks,
        "sha256": digest,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    st.session_state.documents.append(record)
    return True, f"Indexed {len(chunks)} text chunks from {len(pages)} text-readable pages."

def retrieve_context(query: str, subject: str = None, level: str = None, top_k: int = 6):
    candidates = []
    for doc in st.session_state.documents:
        for chunk in doc["chunks"]:
            if subject and subject != "All subjects" and chunk["subject"] != subject:
                continue
            if level and level != "All levels" and chunk["level"] != level:
                continue
            candidates.append(chunk)
    if not candidates:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        corpus = [c["text"] for c in candidates]
        vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=30000
        )
        matrix = vectorizer.fit_transform(corpus + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [dict(item, score=float(score)) for score, item in ranked[:top_k] if score > 0]
    except Exception:
        terms = set(re.findall(r"\w+", query.lower()))
        ranked = []
        for item in candidates:
            body = item["text"].lower()
            score = sum(1 for term in terms if term in body)
            if score:
                ranked.append((score, item))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [dict(item, score=score) for score, item in ranked[:top_k]]

def format_context(chunks):
    blocks = []
    for i, item in enumerate(chunks, start=1):
        blocks.append(
            f"[SOURCE {i}]\nDocument: {item['title']}\n"
            f"Subject: {item['subject']} | Level: {item['level']}\n"
            f"Page: {item['page']} | Type: {item['doc_type']}\n"
            f"Content: {item['text']}"
        )
    return "\n\n".join(blocks)

def grounded_answer(question: str, subject: str, level: str):
    context = retrieve_context(question, subject, level, top_k=6)
    if not context:
        return (
            "I could not find relevant evidence in the documents currently uploaded "
            "to this session. Please upload the relevant textbook, past paper or "
            "marking guide, or broaden the subject/class filters.",
            []
        )
    prompt = f"""
STUDENT QUESTION:
{question}

RETRIEVED SOURCE MATERIAL:
{format_context(context)}

INSTRUCTIONS:
- Answer using ONLY the retrieved source material above.
- Do not introduce unsupported facts or invent figures.
- If the sources are insufficient, say so clearly.
- Explain in clear language suitable for a Ugandan secondary school student.
- Attach source citations to factual claims using [Document title, page X].
- Do not invent page numbers or source titles.
- If sources disagree, identify the disagreement rather than merging them.
"""
    answer = ask_gemini(
        prompt,
        "You are ExamPrep AI, a careful UCE academic tutor. "
        "You must be faithful to the supplied source context."
    )
    return answer, context

# ---------------------------- UI helpers ----------------------------

def metric_card(label, value, note=""):
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
        </div>""",
        unsafe_allow_html=True
    )

def page_header(title, description):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.caption(description)

def require_gemini_notice():
    if not GEMINI_API_KEY:
        st.info(
            "AI features are ready to connect. Add `GEMINI_API_KEY` in "
            "Streamlit Community Cloud → App → Settings → Secrets."
        )

def doc_summary():
    total_chunks = sum(len(d["chunks"]) for d in st.session_state.documents)
    return len(st.session_state.documents), total_chunks

# ---------------------------- Sidebar ----------------------------

with st.sidebar:
    st.markdown(
        "<div style='padding:8px 4px 18px 4px'>"
        "<div style='font-size:1.5rem;font-weight:800;color:white'>🎓 ExamPrep AI</div>"
        "<div style='font-size:.78rem;color:#b9c9e5'>UCE learning and examination platform</div>"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("**YOUR WORKSPACE**")
    nav_items = [
        "Home", "AI Tutor", "Past Papers", "Mock Exams",
        "Revision Centre", "My Performance"
    ]
    if st.session_state.role in ("Teacher", "School Administrator"):
        nav_items += ["Teacher Workspace"]
    if st.session_state.role == "School Administrator":
        nav_items += ["School Administration"]
    selected_page = st.radio(
        "Navigation",
        nav_items,
        index=nav_items.index(st.session_state.page)
        if st.session_state.page in nav_items else 0,
        label_visibility="collapsed",
    )
    st.session_state.page = selected_page
    st.markdown("---")
    st.markdown("**PROFILE**")
    st.session_state.full_name = st.text_input(
        "Display name", value=st.session_state.full_name
    )
    st.session_state.role = st.selectbox(
        "Preview role",
        ["Student", "Teacher", "School Administrator"],
        index=["Student", "Teacher", "School Administrator"].index(
            st.session_state.role
        )
    )
    st.session_state.school_name = st.text_input(
        "School", value=st.session_state.school_name
    )
    st.session_state.class_level = st.selectbox(
        "Class level", LEVELS,
        index=LEVELS.index(st.session_state.class_level)
    )
    st.markdown("---")
    st.caption("Prototype mode · Session data is temporary")
    if GEMINI_API_KEY:
        st.success("Gemini connected")
    else:
        st.warning("Gemini not configured")

# ---------------------------- Home ----------------------------

if st.session_state.page == "Home":
    st.markdown(
        f"""<div class="hero">
        <h1>Welcome back, {st.session_state.full_name}.</h1>
        <p>Prepare smarter, practise confidently and understand your progress.</p>
        </div>""",
        unsafe_allow_html=True
    )
    docs_count, chunks_count = doc_summary()
    completed = len(st.session_state.results)
    avg_score = (
        round(sum(r["percentage"] for r in st.session_state.results) / completed)
        if completed else 0
    )
    cols = st.columns(4)
    with cols[0]: metric_card("Learning resources", docs_count, "Documents in this session")
    with cols[1]: metric_card("Indexed passages", chunks_count, "Searchable text chunks")
    with cols[2]: metric_card("Exams completed", completed, "Practice assessments")
    with cols[3]: metric_card("Average score", f"{avg_score}%", "Based on completed exams")
    st.write("")
    left, right = st.columns([1.45, 1])
    with left:
        st.markdown("### Continue learning")
        actions = [
            ("AI Tutor", "Ask questions using your uploaded academic materials."),
            ("Past Papers", "Find and review questions from uploaded papers."),
            ("Mock Exams", "Create a practice test and assess your knowledge."),
            ("Revision Centre", "Generate concise notes and revision activities."),
        ]
        for title, desc in actions:
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{title}**")
                c1.caption(desc)
                if c2.button("Open", key=f"home_{title}", use_container_width=True):
                    st.session_state.page = title
                    st.rerun()
    with right:
        st.markdown("### Your study profile")
        st.write(f"**School:** {st.session_state.school_name}")
        st.write(f"**Class:** {st.session_state.class_level}")
        st.write(f"**Role:** {st.session_state.role}")
        st.markdown("### Getting started")
        st.markdown(
            "1. Upload a textbook or past paper in **Teacher Workspace**.\n"
            "2. Ask a question in **AI Tutor**.\n"
            "3. Generate a practice examination.\n"
            "4. Review your results and revise weak topics."
        )
        if not st.session_state.documents:
            st.info("No academic documents have been uploaded yet.")

# ---------------------------- AI Tutor ----------------------------

elif st.session_state.page == "AI Tutor":
    page_header("AI Tutor", "Ask questions grounded in the academic documents available to this session.")
    c1, c2 = st.columns(2)
    with c1:
        tutor_subject = st.selectbox("Subject", ["All subjects"] + SUBJECTS, key="tutor_subject")
    with c2:
        tutor_level = st.selectbox("Class", ["All levels"] + LEVELS, index=LEVELS.index(st.session_state.class_level)+1, key="tutor_level")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask an academic question...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            if not GEMINI_API_KEY:
                answer = "Gemini is not configured. Add your GEMINI_API_KEY in Streamlit secrets to enable grounded AI answers."
                st.info(answer)
            else:
                try:
                    with st.spinner("Searching your academic materials and preparing an answer..."):
                        answer, sources = grounded_answer(
                            question,
                            tutor_subject,
                            tutor_level
                        )
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources retrieved"):
                            for item in sources:
                                st.markdown(
                                    f"**{item['title']}** · Page {item['page']} · "
                                    f"{item['subject']} · {item['level']}"
                                )
                                st.caption(item["text"][:900])
                except Exception as exc:
                    answer = f"I couldn't complete the request: {exc}"
                    st.error(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ---------------------------- Past Papers ----------------------------

elif st.session_state.page == "Past Papers":
    page_header("Past Papers Library", "Search and browse the examination papers uploaded to this session.")
    if not st.session_state.documents:
        st.info("No papers or learning documents have been uploaded. Teachers can add them in Teacher Workspace.")
    else:
        f1, f2, f3 = st.columns(3)
        with f1:
            paper_subject = st.selectbox("Subject", ["All subjects"] + SUBJECTS, key="paper_subject")
        with f2:
            paper_level = st.selectbox("Class", ["All levels"] + LEVELS, key="paper_level")
        with f3:
            paper_type = st.selectbox(
                "Document type",
                ["All types", "past_paper", "marking_guide", "textbook", "revision_note"],
                key="paper_type"
            )
        query = st.text_input("Search by topic, question or keyword")
        matching = []
        for doc in st.session_state.documents:
            if paper_subject != "All subjects" and doc["subject"] != paper_subject:
                continue
            if paper_level != "All levels" and doc["level"] != paper_level:
                continue
            if paper_type != "All types" and doc["doc_type"] != paper_type:
                continue
            if query:
                body = " ".join(c["text"] for c in doc["chunks"]).lower()
                if query.lower() not in body and query.lower() not in doc["title"].lower():
                    continue
            matching.append(doc)
        for doc in matching:
            with st.container(border=True):
                a, b = st.columns([4, 1])
                a.markdown(f"**{doc['title']}**")
                a.caption(
                    f"{doc['subject']} · {doc['level']} · {doc['doc_type']} · "
                    f"{doc['pages']} pages"
                )
                if b.button("View details", key=f"view_{doc['sha256']}"):
                    st.session_state.selected_doc = doc["sha256"]
        if not matching:
            st.warning("No matching documents were found.")

        selected = next(
            (d for d in st.session_state.documents
             if d["sha256"] == st.session_state.selected_doc), None
        )
        if selected:
            st.markdown("### Document passages")
            for chunk in selected["chunks"][:30]:
                with st.expander(f"Page {chunk['page']} · Passage {chunk['chunk_index']+1}"):
                    st.write(chunk["text"])

# ---------------------------- Mock Exams ----------------------------

elif st.session_state.page == "Mock Exams":
    page_header("Mock Examinations", "Create a topic-based practice assessment using your uploaded materials.")
    if st.session_state.generated_exam is None:
        with st.form("exam_setup"):
            a, b = st.columns(2)
            with a:
                exam_subject = st.selectbox("Subject", SUBJECTS)
                exam_level = st.selectbox(
                    "Class level", LEVELS,
                    index=LEVELS.index(st.session_state.class_level)
                )
                exam_topic = st.text_input("Topic", placeholder="e.g. Chemical bonding")
            with b:
                exam_count = st.slider("Number of questions", 5, 20, 10)
                exam_difficulty = st.selectbox("Difficulty", ["Easy", "Moderate", "Hard"])
                exam_format = st.selectbox(
                    "Question format",
                    ["Multiple choice", "Mixed (MCQ and short answer)", "Short answer"]
                )
            submitted = st.form_submit_button("Generate practice examination", type="primary", use_container_width=True)
        if submitted:
            if not GEMINI_API_KEY:
                st.error("Add GEMINI_API_KEY to Streamlit secrets to generate an examination.")
            else:
                context = retrieve_context(
                    f"{exam_topic} {exam_subject} {exam_level} examination questions",
                    exam_subject, exam_level, top_k=10
                )
                if not context:
                    st.warning("Upload relevant textbook or examination material first. Questions must be grounded in available sources.")
                else:
                    generation_prompt = f"""
Create a practice examination using ONLY the source material below.
Subject: {exam_subject}
Class: {exam_level}
Topic: {exam_topic}
Number of questions: {exam_count}
Difficulty: {exam_difficulty}
Format: {exam_format}

Return valid JSON only, with this exact structure:
{{
 "title": "string",
 "instructions": "string",
 "questions": [
  {{
   "question": "string",
   "type": "mcq or short",
   "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
   "correct_answer": "string",
   "marking_guide": "string",
   "marks": 1,
   "source_title": "string",
   "source_page": 1
  }}
 ]
}}
For short-answer questions, options must be an empty array.
Do not invent source titles or page numbers. Only use citations present in source material.
Make the questions clear, age-appropriate and non-duplicative.

SOURCE MATERIAL:
{format_context(context)}
"""
                    try:
                        with st.spinner("Generating and validating questions..."):
                            raw = ask_gemini(
                                generation_prompt,
                                "You create careful, curriculum-aligned practice assessments. "
                                "Return valid JSON and never fabricate source references."
                            )
                        match = re.search(r"\{.*\}", raw, flags=re.S)
                        exam = json.loads(match.group(0) if match else raw)
                        if not exam.get("questions"):
                            raise ValueError("No questions were returned.")
                        exam["subject"] = exam_subject
                        exam["level"] = exam_level
                        exam["topic"] = exam_topic
                        exam["created_at"] = datetime.now(timezone.utc).isoformat()
                        st.session_state.generated_exam = exam
                        st.session_state.exam_answers = {}
                        st.session_state.exam_submitted = False
                        st.session_state.exam_result = None
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not generate a valid examination: {exc}")
    else:
        exam = st.session_state.generated_exam
        st.markdown(f"### {exam.get('title', 'Practice Examination')}")
        st.caption(f"{exam['subject']} · {exam['level']} · {exam.get('topic','')} · {len(exam['questions'])} questions")
        st.info(exam.get("instructions", "Answer all questions."))
        if not st.session_state.exam_submitted:
            with st.form("take_exam"):
                answers = {}
                for idx, q in enumerate(exam["questions"]):
                    st.markdown(f"**Question {idx+1} ({q.get('marks', 1)} marks)**")
                    st.write(q["question"])
                    if q.get("type") == "mcq" and q.get("options"):
                        answers[str(idx)] = st.radio(
                            "Select an answer",
                            q["options"],
                            index=None,
                            key=f"exam_q_{idx}",
                            label_visibility="collapsed"
                        )
                    else:
                        answers[str(idx)] = st.text_area(
                            "Your answer",
                            key=f"exam_q_{idx}",
                            label_visibility="collapsed"
                        )
                    st.divider()
                finish = st.form_submit_button("Submit examination", type="primary", use_container_width=True)
            if finish:
                st.session_state.exam_answers = answers
                # Objective questions are marked exactly; written answers are sent for AI marking.
                written = []
                objective_score = 0.0
                objective_total = 0.0
                for idx, q in enumerate(exam["questions"]):
                    maximum = float(q.get("marks", 1))
                    answer = answers.get(str(idx), "")
                    if q.get("type") == "mcq":
                        objective_total += maximum
                        if answer and answer.strip().lower() == str(q.get("correct_answer","")).strip().lower():
                            objective_score += maximum
                    else:
                        written.append({
                            "number": idx+1,
                            "question": q["question"],
                            "student_answer": answer,
                            "marking_guide": q.get("marking_guide", q.get("correct_answer","")),
                            "maximum_marks": maximum
                        })
                ai_marking = []
                if written:
                    if GEMINI_API_KEY:
                        try:
                            mark_prompt = (
                                "Mark the following student responses fairly using ONLY the supplied "
                                "marking guides. Award partial marks where justified. Return JSON only "
                                "as {\"results\":[{\"number\":1,\"awarded_marks\":0,"
                                "\"feedback\":\"...\"}]}. Never exceed maximum marks.\n\n"
                                + json.dumps(written, ensure_ascii=False)
                            )
                            raw_marks = ask_gemini(
                                mark_prompt,
                                "You are a cautious examination marker. Apply the supplied rubric."
                            )
                            m = re.search(r"\{.*\}", raw_marks, flags=re.S)
                            ai_marking = json.loads(m.group(0) if m else raw_marks).get("results", [])
                        except Exception as exc:
                            st.warning(f"AI marking could not be completed: {exc}. Written answers require teacher review.")
                    else:
                        st.warning("Written answers have not been AI-marked because Gemini is not configured.")
                written_score = sum(float(x.get("awarded_marks", 0)) for x in ai_marking)
                written_total = sum(x["maximum_marks"] for x in written)
                total_score = objective_score + written_score
                total_marks = objective_total + written_total
                percentage = round((total_score / total_marks) * 100, 1) if total_marks else 0
                result = {
                    "subject": exam["subject"],
                    "topic": exam.get("topic", ""),
                    "score": total_score,
                    "total": total_marks,
                    "percentage": percentage,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "written_marking": ai_marking,
                    "needs_review": bool(written and not ai_marking),
                }
                st.session_state.exam_result = result
                st.session_state.results.append(result)
                st.session_state.exam_submitted = True
                st.rerun()
        else:
            result = st.session_state.exam_result
            if result:
                st.success(f"Examination submitted. Score: {result['score']} / {result['total']} ({result['percentage']}%)")
                if result["needs_review"]:
                    st.warning("Written responses need teacher marking. The displayed score includes only automatically marked objective questions.")
                for idx, q in enumerate(exam["questions"]):
                    answer = st.session_state.exam_answers.get(str(idx), "")
                    with st.expander(f"Question {idx+1}: Review answer"):
                        st.write(q["question"])
                        st.write(f"**Your answer:** {answer or 'Not answered'}")
                        st.write(f"**Expected answer / marking guide:** {q.get('marking_guide') or q.get('correct_answer')}")
                        if q.get("source_title"):
                            st.caption(f"Source: {q['source_title']}, page {q.get('source_page','not specified')}")
                for mark in result.get("written_marking", []):
                    st.info(f"Written question {mark.get('number')}: {mark.get('awarded_marks')} marks — {mark.get('feedback','')}")
            c1, c2 = st.columns(2)
            if c1.button("Create another examination", use_container_width=True):
                st.session_state.generated_exam = None
                st.session_state.exam_submitted = False
                st.session_state.exam_result = None
                st.rerun()
            if c2.button("Go to performance", use_container_width=True):
                st.session_state.page = "My Performance"
                st.rerun()

# ---------------------------- Revision Centre ----------------------------

elif st.session_state.page == "Revision Centre":
    page_header("Revision Centre", "Generate source-grounded revision notes from uploaded learning materials.")
    r1, r2 = st.columns(2)
    with r1:
        rev_subject = st.selectbox("Subject", SUBJECTS, key="rev_subject")
    with r2:
        rev_level = st.selectbox("Class", LEVELS, index=LEVELS.index(st.session_state.class_level), key="rev_level")
    rev_topic = st.text_input("Topic to revise", placeholder="Enter a topic")
    if st.button("Generate revision notes", type="primary"):
        if not rev_topic.strip():
            st.warning("Enter a topic first.")
        elif not GEMINI_API_KEY:
            require_gemini_notice()
        else:
            sources = retrieve_context(rev_topic, rev_subject, rev_level, top_k=8)
            if not sources:
                st.warning("No supporting materials found. Upload relevant documents first.")
            else:
                prompt = f"""
Create concise, well-organized revision notes for a Ugandan secondary school student.
Subject: {rev_subject}; Level: {rev_level}; Topic: {rev_topic}
Use ONLY the source content below. Include key definitions, explanations, examples,
important points and likely areas for practice only where supported by the sources.
Cite factual sections as [Document title, page X]. Do not invent information.

SOURCE MATERIAL:
{format_context(sources)}
"""
                try:
                    with st.spinner("Preparing revision notes..."):
                        notes = ask_gemini(prompt, "You are a source-grounded academic revision assistant.")
                    st.markdown(notes)
                    st.session_state.revision_notes.append({
                        "topic": rev_topic, "subject": rev_subject,
                        "notes": notes, "date": datetime.now().strftime("%Y-%m-%d")
                    })
                    with st.expander("Retrieved source passages"):
                        for source in sources:
                            st.caption(f"{source['title']} · Page {source['page']}")
                except Exception as exc:
                    st.error(str(exc))
    if st.session_state.revision_notes:
        st.markdown("### Previously generated notes")
        for note in reversed(st.session_state.revision_notes[-10:]):
            with st.expander(f"{note['subject']} — {note['topic']} ({note['date']})"):
                st.markdown(note["notes"])

# ---------------------------- Performance ----------------------------

elif st.session_state.page == "My Performance":
    page_header("My Performance", "Review practice examination results and identify topics for further revision.")
    results = st.session_state.results
    if not results:
        st.info("Complete a mock examination to start building your performance history.")
    else:
        df = pd.DataFrame(results)
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Exams completed", len(results))
        with c2: metric_card("Average score", f"{df['percentage'].mean():.1f}%")
        with c3: metric_card("Best score", f"{df['percentage'].max():.1f}%")
        st.markdown("### Examination history")
        st.dataframe(
            df[["date", "subject", "topic", "score", "total", "percentage"]],
            use_container_width=True,
            hide_index=True
        )
        st.markdown("### Score progression")
        chart_df = df.reset_index().rename(columns={"index": "Exam"})
        st.line_chart(chart_df, x="Exam", y="percentage")
        st.caption("Performance data in this prototype is stored only for the current session.")

# ---------------------------- Teacher Workspace ----------------------------

elif st.session_state.page == "Teacher Workspace":
    page_header("Teacher Workspace", "Upload academic resources and prepare learning materials for students.")
    tab1, tab2 = st.tabs(["Upload documents", "Document library"])
    with tab1:
        st.markdown("#### Add academic material")
        with st.form("upload_document"):
            title = st.text_input("Document title", placeholder="e.g. Senior Four Chemistry Past Paper")
            up_subject = st.selectbox("Subject", SUBJECTS)
            up_level = st.selectbox("Class level", LEVELS)
            up_type = st.selectbox(
                "Document type",
                ["textbook", "past_paper", "marking_guide", "revision_note", "curriculum", "other"]
            )
            uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])
            upload_submit = st.form_submit_button("Upload and index PDF", type="primary", use_container_width=True)
        if upload_submit:
            if not title.strip() or uploaded is None:
                st.error("Enter a document title and select a PDF.")
            else:
                try:
                    with st.spinner("Extracting text and creating searchable passages..."):
                        ok, message = add_document(uploaded, up_subject, up_level, up_type, title.strip())
                    if ok:
                        st.success(message)
                    else:
                        st.warning(message)
                except Exception as exc:
                    st.error(f"Could not process PDF: {exc}")
                    st.caption("Password-protected PDFs and image-only scans may need OCR processing.")
    with tab2:
        docs = st.session_state.documents
        if docs:
            library = pd.DataFrame([{
                "Title": d["title"], "File": d["file_name"],
                "Subject": d["subject"], "Class": d["level"],
                "Type": d["doc_type"], "Pages": d["pages"],
                "Chunks": len(d["chunks"])
            } for d in docs])
            st.dataframe(library, use_container_width=True, hide_index=True)
            remove_title = st.selectbox("Select a document to remove", [d["title"] for d in docs])
            if st.button("Remove selected document"):
                st.session_state.documents = [
                    d for d in docs if d["title"] != remove_title
                ]
                st.success("Removed from the current session.")
                st.rerun()
        else:
            st.info("Your document library is empty.")

# ---------------------------- School Administration ----------------------------

elif st.session_state.page == "School Administration":
    page_header("School Administration", "Overview of this prototype's school workspace.")
    st.warning(
        "This initial Streamlit build uses a role-preview selector, not production "
        "authentication. Do not use it to manage real student records until Supabase "
        "authentication, verified role assignments and row-level security are connected."
    )
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Academic resources", len(st.session_state.documents))
    with c2: metric_card("Practice exams", len(st.session_state.results))
    with c3: metric_card("Active session", "1", "Current browser session")
    st.markdown("### Uploaded resources")
    if st.session_state.documents:
        st.dataframe(pd.DataFrame([{
            "Title": d["title"], "Subject": d["subject"],
            "Level": d["level"], "Type": d["doc_type"]
        } for d in st.session_state.documents]), use_container_width=True, hide_index=True)
    else:
        st.info("No resources have been uploaded in this session.")

# ---------------------------- Footer ----------------------------

st.markdown("---")
st.caption(
    "ExamPrep AI · UCE S1–S4 · Prototype build. "
    "Uploaded files, chat history and results are session-only until persistent storage is connected."
)

