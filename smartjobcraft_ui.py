import os
import time
import logging
from pathlib import Path
import streamlit as st
from openai import OpenAI
import pandas as pd
import json

# Internal imports
from smartjobcraft_day1 import JobParser, SkillNormalizer, score_candidate, Candidate
# Import new function from cv_parser
from cv_parser import extract_skills_and_languages, extract_cv_projects
from smartjobcraft_day1_5 import min_exp, education_bonus # Can be removed if not used
from smartjobcraft_gaprec import PROMPT as GAP_PROMPT, LLM as GAP_LLM
# Import updated functions from smartjobcraft_projectranker
from smartjobcraft_projectranker import rank_projects, compute_project_bonus, extract_projects as ranker_extract_projects
from smartjobcraft_coverletter import generate_cover_letter
import unicodedata

# Logging & Quota Setup
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename="logs/api_calls.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0

def increment_api_calls():
    st.session_state.api_calls += 1
    logging.info(f"API call #{st.session_state.api_calls}")

# LLM Backend & System Prompt
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY environment variable not found!")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a career coach, focus only on career and skill development questions."
}

def chat_llm(messages: list[dict]) -> str:
    increment_api_calls()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

def parse_llm(prompt: str) -> str:
    increment_api_calls()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[SYSTEM_PROMPT, {"role":"user","content":prompt}],
        temperature=0,
        response_format={"type":"json_object"},
    )
    return resp.choices[0].message.content

# Sidebar Chat Bot
st.sidebar.title("🎙 Career Coach")
if "history" not in st.session_state:
    st.session_state.history = []
user_q = st.sidebar.text_input("Ask a question:", "")
if user_q:
    st.session_state.history.append(("user", user_q))
    # Add SYSTEM_PROMPT at the beginning of the message list
    messages_for_llm = [SYSTEM_PROMPT] + [{"role": r, "content": m} for r, m in st.session_state.history]
    assistant_a = chat_llm(messages_for_llm)
    st.session_state.history.append(("assistant", assistant_a))

for role, msg in st.session_state.history:
    if role == "user":
        st.sidebar.markdown(f"**You:** {msg}")
    elif role == "assistant": # Don't show system messages
        st.sidebar.markdown(f"**Bot:** {msg}")

if st.sidebar.button("👍 This response was helpful"):
    logging.info("User marked last response as helpful")
if st.sidebar.button("👎 This response was not helpful"):
    logging.info("User marked last response as unhelpful")
st.sidebar.markdown(f"**API calls:** {st.session_state.api_calls}")

# Main Page
st.title("SmartJob Craft 🚀")

# CV Upload
st.header("1. Upload Your CV")
cv_file = st.file_uploader("Upload CV in PDF format", type=["pdf"])
# Initialize session state variables
if 'cand_skills' not in st.session_state:
    st.session_state.cand_skills = set()
if 'cand_languages' not in st.session_state:
    st.session_state.cand_languages = set()
if 'cand_cv_projects' not in st.session_state:
    st.session_state.cand_cv_projects = []

if cv_file:
    # Save CV file to a temporary path
    temp_cv_path = Path("tmp_cv.pdf")
    with open(temp_cv_path, "wb") as f:
        f.write(cv_file.getbuffer())
    st.success("CV uploaded and being parsed...")
    
    # Initialize SkillNormalizer (with default CSV path)
    try:
        normalizer = SkillNormalizer() # Default "tests/skills_variants.csv"
    except FileNotFoundError:
        st.error("Skill normalizer file 'tests/skills_variants.csv' not found.")
        st.stop()
        
    # Extract skills and languages
    st.session_state.cand_skills, st.session_state.cand_languages = extract_skills_and_languages(temp_cv_path, normalizer)
    # Extract projects (use extract_projects from project_ranker)
    st.session_state.cand_cv_projects = ranker_extract_projects(str(temp_cv_path)) 

    st.write("**Skills Found:**", ", ".join(sorted(st.session_state.cand_skills)) or "None")
    st.write("**Languages Found:**", ", ".join(sorted(st.session_state.cand_languages)) or "None")
    if st.session_state.cand_cv_projects and st.session_state.cand_cv_projects != ["No specific projects extracted, check regex or CV structure."]:
        st.write("**Projects Found:**")
        for proj in st.session_state.cand_cv_projects:
            st.markdown(f"- {proj[:150]}...") # First 150 chars of each project
    else:
        st.warning("No projects could be extracted from the CV or projects are not in the expected format.")
else:
    st.info("Please upload your CV.")

# Job Ad Input
st.header("2. Paste Your Job Ad")
job_ad = st.text_area("Paste the job description here", height=200)

# Matching & Scoring
if st.button("🔍 Match and Score"):
    if not cv_file or not job_ad.strip():
        st.error("Please both upload a CV and enter a job description.")
    elif not st.session_state.cand_skills and not st.session_state.cand_languages:
        st.error("No skills or languages could be extracted from the CV. Please check your CV.")
    else:
        # Reload the normalizer (already done in CV upload but included here too)
        try:
            normalizer = SkillNormalizer()
        except FileNotFoundError:
            st.error("Skill normalizer file 'tests/skills_variants.csv' not found.")
            st.stop()

        parser = JobParser(parse_llm)
        try:
            job = parser.parse(job_ad)
        except Exception as e:
            st.error(f"Failed to parse job description: {e}")
            st.stop()
        
        # Normalize must-have and nice-to-have skills
        must_skills_job = set(normalizer.canonical(s) for s in job.must_have)
        nice_skills_job = set(normalizer.canonical(s) for s in job.nice_to_have)
        
        # Combine candidate's skills and languages
        combined_candidate_skills = st.session_state.cand_skills.union(st.session_state.cand_languages)
        
        # Calculate score (initial score without exp_bonus and project_bonus)
        score = score_candidate(combined_candidate_skills, job) 
        
        st.subheader("Matching Results")
        st.write(f"**Position:** {job.title} @ {job.company}")
        st.write(f"**Score (Base):** {score:.2f}")
        st.write("**Required Must-have Skills:**", ", ".join(sorted(must_skills_job)) or "None")
        st.write("**Desired Nice-to-have Skills:**", ", ".join(sorted(nice_skills_job)) or "None")
        
        # Calculate skill gaps
        all_job_skills = must_skills_job.union(nice_skills_job)
        gaps = sorted(all_job_skills - combined_candidate_skills)
        st.write("**Skill Gaps:**", ", ".join(gaps) or "Congratulations, no skill gaps found!")
        
        # Micro-Learning Plan
        if score < 0.6 and gaps:
            st.subheader("📚 Micro-Learning Plan")
            # Format GAP_PROMPT correctly
            micro_learning_prompt = GAP_PROMPT.format(title=job.title, missing=", ".join(gaps))
            try:
                plan_json_str = GAP_LLM(micro_learning_prompt)
                plan_data = json.loads(plan_json_str)
                plan_markdown = plan_data.get("plan", "*Failed to create plan.*")
                st.markdown(plan_markdown)
            except json.JSONDecodeError:
                st.error("JSON format error while generating micro-learning plan.")
            except Exception as e:
                st.error(f"Error generating micro-learning plan: {e}")
        
        # Project Ranking
        st.subheader("📁 Project Ranking")
        # Create Candidate object
        candidate_profile = Candidate(
            name="Candidate", # Real name could be extracted from CV
            headline="", # Could be extracted from CV
            skills=combined_candidate_skills,
            projects=st.session_state.cand_cv_projects, # Projects extracted from CV
            education=[] # Could be extracted from CV
        )
        
        top_project, coverage = rank_projects(job, candidate_profile, normalizer)
        project_bonus_val = compute_project_bonus(coverage)
        final_score = score + project_bonus_val # Add project bonus to score (exp_bonus could also be added)
        
        st.write(f"**Most Relevant Project:** {top_project or 'No suitable project found.'}")
        st.write(f"Project Coverage Ratio: {coverage:.2f}, Project Bonus: {project_bonus_val:.2f}")
        st.write(f"**Final Score (with Project Bonus):** {final_score:.2f}")

        # Cover Letter Generation
        st.subheader("📝 Generate Cover Letter")
        cover_letter_meta = {
            "job_title": job.title,
            "company": job.company,
            "must_skills": list(must_skills_job), # Normalized must-have skills
            "top_project": top_project or "", # Empty string if no project
        }
        try:
            # generate_cover_letter expects client parameter
            letter = generate_cover_letter(cover_letter_meta, client) 
            st.text_area("Generated Cover Letter", letter, height=300)
        except Exception as e:
            st.error(f"Failed to generate cover letter: {e}")
            letter = ""
        
        # PDF Download (if letter exists)
        if letter:
            st.subheader("📄 Download Cover Letter")
            from fpdf import FPDF
            def clean_text_for_pdf(text):
                # Simple cleaning, additional processing might be needed for complex characters
                return unicodedata.normalize('NFKD', text).encode('latin-1', 'ignore').decode('latin-1')
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            cleaned_letter = clean_text_for_pdf(letter)
            for line in cleaned_letter.split("\n"):
                pdf.multi_cell(0, 10, line)
            
            pdf_output_path = "coverletter.pdf"
            try:
                pdf.output(pdf_output_path, "F") # "F" saves to disk
                with open(pdf_output_path, "rb") as fp:
                    st.download_button(
                        label="📥 Download Cover Letter as PDF",
                        data=fp,
                        file_name="coverletter.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.error(f"Failed to create or save PDF: {e}")

        # Download results as CSV
        summary_data = [{
            "file_id": "current_session",
            "title": job.title,
            "score": round(final_score, 3),
            "gaps": "; ".join(gaps),
            "top_project": top_project or "N/A",
            "proj_cov": round(coverage, 3),
        }]
        summary_df = pd.DataFrame(summary_data)
        st.download_button(
            label="📥 Download Results as CSV",
            data=summary_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="match_summary.csv",
            mime="text/csv"
        )
