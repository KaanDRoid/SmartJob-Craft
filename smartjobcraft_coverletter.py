# smartjobcraft_coverletter.py
# This module generates tailored cover letters for job applications based on job metadata and candidate's relevant projects.
# It leverages OpenAI's API to create personalized and concise cover letters addressing skill gaps.

from __future__ import annotations
import os, json, re
from pathlib import Path
from typing import Dict

from openai import OpenAI

# --- CONFIG ---
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("🚨 OPENAI_API_KEY environment variable is not set!")
client = OpenAI(api_key=OPENAI_KEY)

# Directory for input job JSON files (ensure this is the correct directory you intend to use)
JOB_JSON_DIR   = Path("job_json2") 
# Path to the summary CSV file (ensure this is the correct file)
SUMMARY_CSV    = Path("summary2.csv") 
# Directory where generated cover letters will be saved
OUTPUT_DIR     = Path("cover_letters")
OUTPUT_DIR.mkdir(exist_ok=True) # Create the output directory if it doesn't exist

# --- Prompt Template ---
# This template is used to instruct the LLM on how to generate the cover letter.
# It includes placeholders like {job_title}, {company}, {top_project}, and {must_skills}
# which will be filled in with specific details for each job.
COVER_TMPL = """
You are a seasoned career coach and professional writer.
Write a concise, engaging cover letter for the position of **{job_title}** at **{company}**.

Requirements:
- Open with a personalized salutation (e.g., "Dear Hiring Manager at [Company Name],").
- Mention the candidate's most relevant project: _"{top_project}"_ and briefly explain its relevance if possible.
- Highlight how the candidate's skills in {must_skills} align with the job's must-have requirements.
- Keep the letter under 200 words.
- Maintain a professional yet friendly and enthusiastic tone.
- Conclude with a call to action, expressing eagerness to discuss qualifications further.

Return only the cover letter text (no JSON, no markdown fences, no introductory phrases like "Here is the cover letter:").
"""

# --- Helpers ---
def load_job_metadata() -> Dict[str, Dict]:
    """
    Reads the summary CSV (e.g., summary2.csv) and corresponding JSON files 
    from the job JSON directory (e.g., job_json2/) to collect metadata for each job.

    For each relevant job (score < 0.6 and a top project identified), it gathers:
      - job_title: The title of the job.
      - company: The name of the company.
      - must_skills: A list of essential skills required for the job.
      - top_project: The candidate's most relevant project for this job.

    Returns:
        A dictionary where keys are file_ids (from the summary CSV) and values are
        dictionaries containing the extracted metadata for cover letter generation.
    """
    import pandas as pd
    try:
        df = pd.read_csv(SUMMARY_CSV, encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"Error: The summary file {SUMMARY_CSV} was not found.")
        return {}
    except Exception as e:
        print(f"Error reading {SUMMARY_CSV}: {e}")
        return {}

    data = {}
    for row in df.to_dict(orient="records"):
        fid = row.get("file_id")
        score = row.get("score")
        top_project = row.get("top_project")

        if fid is None or score is None:
            print(f"Warning: Skipping row due to missing 'file_id' or 'score': {row}")
            continue

        # Generate cover letters only for jobs that are a reasonable match (score < 0.6 implies a gap to address)
        # and where a relevant top project has been identified.
        # Adjust the score threshold as needed. A lower score might indicate a larger gap,
        # making a tailored cover letter more crucial.
        if score < 0.6 and top_project: # Ensure top_project is not None or empty
            json_file_path = JOB_JSON_DIR / f"{fid}.json"
            try:
                job_json_content = json.loads(json_file_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                print(f"Warning: JSON file not found for {fid} at {json_file_path}. Skipping.")
                continue
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON for {fid} at {json_file_path}. Skipping.")
                continue
            except Exception as e:
                print(f"Warning: Error reading JSON file for {fid} at {json_file_path}: {e}. Skipping.")
                continue
            
            data[fid] = {
                "job_title": job_json_content.get("title", job_json_content.get("job_title", "[Job Title Not Found]")),
                "company":   job_json_content.get("company", "[Company Name Not Found]"),
                "must_skills": job_json_content.get("must_have", []),
                "top_project": top_project,
            }
    return data

def format_skills(skills: list[str]) -> str:
    """Formats a list of skills into a human-readable string. 
    Uses an Oxford comma for lists of three or more skills.
    Example: ["Python", "SQL", "Pandas"] -> "Python, SQL, and Pandas"
    """
    if not skills: # Handle empty list
        return "a range of relevant skills"
    if len(skills) == 1:
        return skills[0]
    if len(skills) == 2:
        return skills[0] + " and " + skills[1]
    # For three or more skills, use an Oxford comma
    return ", ".join(skills[:-1]) + ", and " + skills[-1]

def generate_cover_letter(meta: dict, llm_client) -> str:
    """Generates a cover letter using the OpenAI API based on the provided metadata."""
    prompt = COVER_TMPL.format(
        job_title    = meta.get("job_title", "[Job Title]"),
        company      = meta.get("company", "[Company]"),
        top_project  = meta.get("top_project", "[Relevant Project]"),
        must_skills  = format_skills(meta.get("must_skills", [])),
    )
    try:
        resp = llm_client.chat.completions.create(
            model="gpt-4o-mini", # Using a cost-effective and capable model
            messages=[{"role":"user","content":prompt}],
            temperature=0.3, # Lower temperature for more focused and less random output
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during OpenAI API call for job '{meta.get('job_title')}': {e}")
        return "[Error generating cover letter]"

# --- Main ---
def main():
    """Main function to load job metadata and generate cover letters."""
    # Initialize the OpenAI client locally within main or ensure the global one is used if preferred.
    # Using a local client here to be explicit.
    local_client = OpenAI(api_key=OPENAI_KEY) 

    print("Loading job metadata to generate cover letters...")
    metas = load_job_metadata()
    if not metas:
        print("🚫 No suitable job postings (score < 0.6 and top_project identified) were found to generate cover letters for.")
        return

    print(f"Found {len(metas)} job(s) eligible for cover letter generation.")
    for fid, meta in metas.items():
        print(f"Generating cover letter for: {meta.get('job_title', fid)} at {meta.get('company', 'N/A')} (File ID: {fid})")
        letter = generate_cover_letter(meta, local_client)
        
        if "[Error generating cover letter]" in letter:
            print(f"❌ Failed to generate cover letter for {fid}.")
            continue # Skip saving if there was an error

        out_path = OUTPUT_DIR / f"{fid}.md"
        try:
            out_path.write_text(letter, encoding="utf-8")
            print(f"✅ Cover letter for {fid} saved to {out_path} ({len(letter.split())} words)")
        except Exception as e:
            print(f"❌ Error saving cover letter for {fid} to {out_path}: {e}")

if __name__ == "__main__":
    main()
