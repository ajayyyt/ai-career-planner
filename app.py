# AI Career Planner: career dashboard + coding corner
# Setup: pip install streamlit google-genai pydantic pandas altair
# Put your key in .streamlit/secrets.toml as  GEMINI_API_KEY = "..."  (or set the env var).

import json
import os
import time
from typing import List

import altair as alt
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field

st.set_page_config(page_title="AI Career Planner", page_icon="🎯", layout="centered")

MODEL = "gemini-3.5-flash-lite"

# NOTE: replace this with your own SYSTEM_PROMPT (keep the role and rules you wrote).
SYSTEM_PROMPT = """You are a warm, practical career mentor for students and job seekers in India.
Rules:
- Be realistic and honest. Never promise jobs or salaries.
- Suggest only roles that fit the person's skills, interests, education and goal.
- Use free learning resources only (name the exact course, channel or site).
- Salaries are indicative ranges in LPA (lakhs per annum) for India.
- Fit tasks and projects to the hours per week the person can study.
- If a resume is attached, use it to understand their real skills and experience.
- Use simple, plain English. Be specific, not generic."""


# ---------- Structured answer (schema for Gemini) ----------
class Role(BaseModel):
    title: str
    why_it_fits: str
    match_score: int = Field(ge=0, le=100)
    salary_range_lpa: str


class SkillGap(BaseModel):
    skill: str
    current_level: int = Field(ge=0, le=10)
    needed_level: int = Field(ge=0, le=10)
    free_resource: str


class Phase(BaseModel):
    tasks: List[str] = Field(min_length=3, max_length=3)
    mini_project: str


class Roadmap(BaseModel):
    days_1_30: Phase
    days_31_60: Phase
    days_61_90: Phase


class CareerPlan(BaseModel):
    roles: List[Role] = Field(min_length=3, max_length=3)
    skill_gaps: List[SkillGap] = Field(min_length=4, max_length=6)
    roadmap: Roadmap
    todays_action: str


# ---------- Gemini helpers ----------
def get_client():
    from google import genai

    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    key = key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No GEMINI_API_KEY found. Add it to .streamlit/secrets.toml.")
    return genai.Client(api_key=key)


FALLBACK_MODELS = ["gemini-2.5-flash"]  # used if the main model stays overloaded


def generate_with_retry(client, model=MODEL, **kwargs):
    """Call Gemini; retry on 503/429 with short waits, then try fallback models."""
    last_error = None
    for m in [model] + FALLBACK_MODELS:
        for attempt in range(3):
            try:
                return client.models.generate_content(model=m, **kwargs)
            except Exception as e:
                last_error = e
                msg = str(e)
                if not any(c in msg for c in ("503", "429", "UNAVAILABLE", "overloaded")):
                    raise
                time.sleep(2 ** attempt)  # waits 1s, 2s, 4s
    raise last_error


def build_gemini_contents(answers, resume_bytes):
    """Resume PDF (if any) + form answers. Everything stays in memory."""
    from google.genai import types

    contents = []
    if resume_bytes:
        contents.append(
            types.Part.from_bytes(data=resume_bytes, mime_type="application/pdf")
        )
    contents.append(
        "Make a career plan for this person:\n" + json.dumps(answers, indent=2)
    )
    return contents


def generate_plan(answers, resume_bytes):
    from google.genai import types

    client = get_client()
    response = generate_with_retry(
        client,
        model=MODEL,
        contents=build_gemini_contents(answers, resume_bytes),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=CareerPlan,
            temperature=0.7,
        ),
    )
    return CareerPlan.model_validate_json(response.text).model_dump()


def answer_followup(question, plan, answers, history):
    from google.genai import types

    client = get_client()
    past = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    prompt = (
        f"Person's profile:\n{json.dumps(answers)}\n\n"
        f"Their career plan (JSON):\n{json.dumps(plan)}\n\n"
        f"Conversation so far:\n{past}\n\n"
        f"New question: {question}"
    )
    response = generate_with_retry(
        client,
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                SYSTEM_PROMPT
                + "\nYou are answering follow-up questions about the plan below. "
                "Use the plan as context. Answer in under 150 words."
            ),
            temperature=0.6,
        ),
    )
    return response.text.strip()


def plan_to_markdown(plan, name):
    lines = [f"# Career plan for {name}", "", "## Do this today", plan["todays_action"], ""]
    lines += ["## Roles that fit you", ""]
    for r in plan["roles"]:
        lines += [
            f"### {r['title']} ({r['match_score']}/100 match)",
            r["why_it_fits"],
            f"*Indicative salary: {r['salary_range_lpa']} LPA*",
            "",
        ]
    lines += ["## Skill gaps", ""]
    for s in plan["skill_gaps"]:
        lines.append(
            f"- **{s['skill']}**: {s['current_level']}/10 now, need {s['needed_level']}/10. "
            f"Free resource: {s['free_resource']}"
        )
    lines += ["", "## 90-day roadmap", ""]
    for key, label in [
        ("days_1_30", "Days 1-30"),
        ("days_31_60", "Days 31-60"),
        ("days_61_90", "Days 61-90"),
    ]:
        phase = plan["roadmap"][key]
        lines.append(f"### {label}")
        lines += [f"- [ ] {t}" for t in phase["tasks"]]
        lines += [f"\n**Mini project:** {phase['mini_project']}", ""]
    lines.append("*AI suggestions, not guarantees.*")
    return "\n".join(lines)


# ---------- Coding helper (Q&A + practice questions) ----------
CODING_PROMPT = (
    "You are a friendly coding tutor. Answer simple programming questions clearly "
    "with a short explanation and a small code example. Keep answers under 200 words. "
    "If the question is not about coding, politely say you can only help with coding."
)


def ask_coding(contents, system_prompt=CODING_PROMPT):
    from google.genai import types

    client = get_client()
    response = generate_with_retry(
        client,
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt, temperature=0.6
        ),
    )
    return (response.text or "").strip()


def coding_helper():
    st.subheader("💻 Coding corner")
    ask_tab, practice_tab = st.tabs(["💬 Ask a coding question", "🧩 Practice questions"])

    # --- Ask a question ---
    with ask_tab:
        if "ch_history" not in st.session_state:
            st.session_state.ch_history = []

        for m in st.session_state.ch_history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        q = st.text_input(
            "Your coding question",
            key="ch_question",
            placeholder="e.g. What is the difference between a list and a tuple?",
        )
        c1, c2 = st.columns(2)
        ask = c1.button("Ask", key="ch_ask", use_container_width=True)
        if c2.button("Clear answers", key="ch_clear", use_container_width=True):
            st.session_state.ch_history = []
            st.rerun()

        if ask and q.strip():
            try:
                with st.spinner("Thinking…"):
                    answer = ask_coding(q.strip())
            except Exception as e:
                answer = f"Sorry, I couldn't answer that. Please try again. ({e})"
            st.session_state.ch_history.append({"role": "user", "content": q.strip()})
            st.session_state.ch_history.append({"role": "assistant", "content": answer})
            st.rerun()

    # --- Practice questions ---
    with practice_tab:
        if "ch_practice" not in st.session_state:
            st.session_state.ch_practice = {}

        lang_col, level_col = st.columns(2)
        with lang_col:
            language = st.text_input("Language / topic", value="Python", key="ch_lang")
        with level_col:
            level = st.selectbox(
                "Level", ["Beginner", "Intermediate", "Advanced"], key="ch_level"
            )

        if st.button("Generate 5 questions", key="ch_generate", use_container_width=True):
            prompt = (
                f"Create exactly 5 {level}-level coding practice questions in {language}. "
                "Number them 1 to 5. Each one is a short problem statement with a "
                "one-line example input and output. Do not include solutions."
            )
            try:
                with st.spinner("Generating…"):
                    st.session_state.ch_practice[f"{language}|{level}"] = ask_coding(
                        prompt, "You write clear, well-scoped coding practice problems."
                    )
            except Exception as e:
                st.error(f"Could not generate questions. Please try again. ({e})")

        questions = st.session_state.ch_practice.get(f"{language}|{level}")
        if questions:
            st.markdown(questions)

            n_col, hint_col, sol_col = st.columns([1, 1, 1])
            with n_col:
                q_num = st.number_input(
                    "Question #", min_value=1, max_value=5, step=1, key="ch_qnum"
                )
            for col, label, mode in [
                (hint_col, "💡 Hint", "a short hint only, without giving the answer"),
                (sol_col, "✅ Solution", "a full solution with a short explanation"),
            ]:
                if col.button(label, key=f"ch_btn_{label}", use_container_width=True):
                    try:
                        with st.spinner("Working…"):
                            reply = ask_coding(
                                f"Practice questions:\n{questions}\n\n"
                                f"Give {mode} for question {int(q_num)}."
                            )
                        st.markdown(reply)
                    except Exception as e:
                        st.error(f"Sorry, that didn't work. ({e})")


# ---------- Page style ----------
st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] { max-width: 1000px; padding-top: 2.5rem; }
    .hero { text-align: center; margin-bottom: 1.5rem; }
    .hero h1 { font-size: 2.6rem; margin-bottom: 0.2rem; }
    .hero p { color: #475569; font-size: 1.1rem; margin: 0; }
    .step { color: #2563EB; font-weight: 600; font-size: 0.85rem;
            letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.3rem; }
    .score { font-size: 2.4rem; font-weight: 700; color: #2563EB; line-height: 1.1; }
    .score span { font-size: 1rem; font-weight: 500; color: #475569; }
    .footer { text-align: center; color: #64748B; font-size: 0.85rem; padding: 0.5rem 0 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🎯 AI Career Planner</h1>
      <p>My first AI product, built by <b>AJAY</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Step 1 · Resume (optional) ----------
with st.container(border=True):
    st.subheader("Step 1 · Resume (optional)")
    resume_file = st.file_uploader(
        "Upload your resume as a PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )
    st.caption(
        "Your resume is sent to Google Gemini to make your plan and is not stored by "
        "this app. On Gemini's free tier Google may use it to improve its products."
    )

has_resume = resume_file is not None

# ---------- Step 2 · About you ----------
with st.container(border=True):
    st.subheader("Step 2 · About you")

    name_col, education_col = st.columns(2)
    with name_col:
        name = st.text_input("Your name")
    with education_col:
        education = st.selectbox(
            "Highest education",
            [
                "Plus Two / 12th",
                "Diploma",
                "B.Tech / B.E.",
                "B.Sc / BCA",
                "B.Com / BBA",
                "BA",
                "Master's degree",
                "Other",
            ],
        )

    st.radio(
        "Where are you right now?",
        [
            "Student",
            "Graduate looking for a job",
            "Working, want to switch",
            "Career break",
        ],
        horizontal=True,
        key="current_situation",
    )

    skills_col, interests_col = st.columns(2)
    with skills_col:
        skills = st.text_area(
            "Skills you already have (optional)"
            if has_resume
            else "Skills you already have"
        )
    with interests_col:
        interests = st.text_area("What do you enjoy?")

    goal = st.text_input("Your goal")
    hours = st.slider("Hours per week you can study", min_value=2, max_value=40, value=10)

    if st.button("🚀 Plan my career", type="primary", use_container_width=True):
        missing = []
        if not name.strip():
            missing.append("Your name")
        if not has_resume and not skills.strip():
            missing.append("Skills you already have")
        if not goal.strip():
            missing.append("Your goal")

        if missing:
            st.warning("Please fill in: " + ", ".join(missing))
        else:
            answers = {
                "name": name.strip(),
                "highest_education": education,
                "current_situation": st.session_state.current_situation,
                "skills": skills.strip(),
                "interests": interests.strip(),
                "goal": goal.strip(),
                "hours_per_week": hours,
                "resume_uploaded": has_resume,
            }
            resume_bytes = resume_file.getvalue() if has_resume else None  # memory only
            try:
                with st.spinner("Building your plan… this takes a few seconds"):
                    plan = generate_plan(answers, resume_bytes)
                # New plan: reset ticked tasks and chat
                for k in [k for k in st.session_state if k.startswith("task_")]:
                    del st.session_state[k]
                st.session_state.answers = answers
                st.session_state.plan = plan
                st.session_state.chat = []
            except Exception as e:
                st.error(f"Could not make your plan. Please try again. ({e})")

# ---------- Dashboard ----------
if "plan" in st.session_state:
    plan = st.session_state.plan
    answers = st.session_state.answers

    # Do this today
    st.success(f"### ✅ Do this today\n{plan['todays_action']}")

    # Role cards
    st.subheader("Roles that fit you")
    cols = st.columns(3)
    for col, role in zip(cols, plan["roles"]):
        score = max(0, min(100, role["match_score"]))
        with col:
            with st.container(border=True):
                st.markdown(f"**{role['title']}**")
                st.markdown(
                    f'<div class="score">{score}<span> / 100 match</span></div>',
                    unsafe_allow_html=True,
                )
                st.progress(score / 100)
                st.write(role["why_it_fits"])
                st.caption(f"Indicative salary: {role['salary_range_lpa']} LPA")

    # Skill gaps chart
    st.subheader("Your skill gaps")
    gaps = plan["skill_gaps"]
    df = pd.DataFrame(
        [
            {"skill": g["skill"], "level": g["current_level"], "type": "Current"}
            for g in gaps
        ]
        + [
            {"skill": g["skill"], "level": g["needed_level"], "type": "Needed"}
            for g in gaps
        ]
    )
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("skill:N", title=None, sort=[g["skill"] for g in gaps]),
            yOffset=alt.YOffset("type:N", sort=["Current", "Needed"]),
            x=alt.X("level:Q", title="Level (0-10)", scale=alt.Scale(domain=[0, 10])),
            color=alt.Color(
                "type:N",
                title=None,
                scale=alt.Scale(
                    domain=["Current", "Needed"], range=["#94A3B8", "#2563EB"]
                ),
            ),
        )
        .properties(height=max(200, 60 * len(gaps)))
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("**Free resources**")
    for g in gaps:
        st.markdown(f"- **{g['skill']}**: {g['free_resource']}")

    # Roadmap
    st.subheader("Your 90-day roadmap")
    phases = [
        ("days_1_30", "Days 1-30"),
        ("days_31_60", "Days 31-60"),
        ("days_61_90", "Days 61-90"),
    ]
    tabs = st.tabs([label for _, label in phases])
    for tab, (key, label) in zip(tabs, phases):
        phase = plan["roadmap"][key]
        with tab:
            for i, task in enumerate(phase["tasks"]):
                st.checkbox(task, key=f"task_{key}_{i}")
            st.info(f"🛠️ **Mini project:** {phase['mini_project']}")

    # Coding corner (new)
    st.divider()
    coding_helper()

    # Download
    st.download_button(
        "⬇️ Download my plan",
        data=plan_to_markdown(plan, answers["name"]),
        file_name="my_career_plan.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # ---------- Chat ----------
    st.subheader("Ask about your plan")
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    question = st.chat_input("Ask about your plan")
    if question:
        with st.chat_message("user"):
            st.write(question)
        history = list(st.session_state.chat)
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking…"):
                    reply = answer_followup(question, plan, answers, history)
            except Exception as e:
                reply = f"Sorry, I couldn't answer that. Please try again. ({e})"
            st.write(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})

# Footer
st.divider()
st.markdown(
    '<div class="footer">Built at the Techolas AI Wing weekend · AI suggestions, not guarantees.</div>',
    unsafe_allow_html=True,
)