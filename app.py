# # CHECKPOINT 1 — "Hello, AI builder"
# # This is the file every attendee starts with. If you can see this page
# # in the preview on the right, your setup is done.
# # It uses the same page layout as the final app, so the look never changes.

# import streamlit as st

# st.set_page_config(page_title="AI Career Planner", page_icon="🎯", layout="centered")

# # A little CSS (page styling). "centered" layout keeps everything in a middle
# # column; we make that column a bit wider so the 3 role cards fit side by side.
# st.markdown(
#     """
#     <style>
#     [data-testid="stMainBlockContainer"] { max-width: 1000px; padding-top: 2.5rem; }
#     .hero { text-align: center; margin-bottom: 1.5rem; }
#     .hero h1 { font-size: 2.6rem; margin-bottom: 0.2rem; }
#     .hero p { color: #475569; font-size: 1.1rem; margin: 0; }
#     .step { color: #2563EB; font-weight: 600; font-size: 0.85rem;
#             letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.3rem; }
#     .score { font-size: 2.4rem; font-weight: 700; color: #2563EB; line-height: 1.1; }
#     .score span { font-size: 1rem; font-weight: 500; color: #475569; }
#     .footer { text-align: center; color: #64748B; font-size: 0.85rem; padding: 0.5rem 0 1rem; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # Centered title and subtitle
# st.markdown(
#     """
#     <div class="hero">
#       <h1>🎯 AI Career Planner</h1>
#       <p>My first AI product, built by <b>AJAY</b>.</p>
#     </div>
#     """,
#     unsafe_allow_html=True,
    
# )

# with st.container(border=True):
#     st.subheader("About you")

#     name_col, education_col = st.columns(2)
#     with name_col:
#         name = st.text_input("Your name")
#     with education_col:
#         education = st.selectbox(
#             "Highest education",
#             [
#                 "Plus Two / 12th",
#                 "Diploma",
#                 "B.Tech / B.E.",
#                 "B.Sc / BCA",
#                 "B.Com / BBA",
#                 "BA",
#                 "Master's degree",
#                 "Other",
#             ],
#         )

#     st.radio(
#         "Where are you right now?",
#         [
#             "Student",
#             "Graduate looking for a job",
#             "Working, want to switch",
#             "Career break",
#         ],
#         horizontal=True,
#         key="current_situation",
#     )

#     skills_col, interests_col = st.columns(2)
#     with skills_col:
#         skills = st.text_area("Skills you already have")
#     with interests_col:
#         interests = st.text_area("What do you enjoy?")

#     goal = st.text_input("Your goal")
#     hours = st.slider("Hours per week you can study", min_value=2, max_value=40, value=10)

#     if st.button("🚀 Plan my career", type="primary", use_container_width=True):
#         missing = []
#         if not name.strip():
#             missing.append("Your name")
#         if not skills.strip():
#             missing.append("Skills you already have")
#         if not goal.strip():
#             missing.append("Your goal")

#         if missing:
#             st.warning("Please fill in: " + ", ".join(missing))
#         else:
#             answers = {
#                 "name": name.strip(),
#                 "highest_education": education,
#                 "current_situation": st.session_state.current_situation,
#                 "skills": skills.strip(),
#                 "interests": interests.strip(),
#                 "goal": goal.strip(),
#                 "hours_per_week": hours,
#             }
#             st.session_state.answers = answers

# if "answers" in st.session_state:
#     st.subheader("What the AI will receive")
#     st.json(st.session_state.answers)

# # Footer
# st.divider()
# st.markdown(
#     '<div class="footer">Built at the Techolas AI Wing weekend · AI suggestions, not guarantees.</div>',
#     unsafe_allow_html=True,
# )






# AI Career Planner: career dashboard
# Setup: pip install streamlit google-genai pydantic pandas altair
# Put your key in .streamlit/secrets.toml as  GEMINI_API_KEY = "..."  (or set the env var).

import json
import os
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
    response = client.models.generate_content(
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
    response = client.models.generate_content(
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