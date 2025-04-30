# smartjobcraft_coverletter.py

from __future__ import annotations
import os, json, re
from pathlib import Path
from typing import Dict

from openai import OpenAI

# --- CONFIG ---
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("🚨 OPENAI_API_KEY ortam değişkeni tanımlı değil!")
client = OpenAI(api_key=OPENAI_KEY)

JOB_JSON_DIR   = Path("job_json2")
SUMMARY_CSV    = Path("summary2.csv")
OUTPUT_DIR     = Path("cover_letters")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Prompt Template ---
COVER_TMPL = """
You are a seasoned career coach and professional writer.
Write a concise, engaging cover letter for the position of **{job_title}** at **{company}**.

Requirements:
- Open with a personalized salutation.
- Mention your top relevant project: _"{top_project}"_.
- Highlight how your skills {must_skills} match their must-haves.
- Keep it under 200 words.
- Use a professional yet friendly tone.

Return only the cover letter text (no JSON, no markdown fences).
"""

# --- Helpers ---
def load_job_metadata() -> Dict[str, Dict]:
    """
    Reads summary2.csv and job_json2/*.json to collect for each file_id:
      - job_title, company
      - must_have skills
      - top_project
    """
    import pandas as pd
    df = pd.read_csv(SUMMARY_CSV, encoding="utf-8-sig")
    data = {}
    for row in df.to_dict(orient="records"):
        fid = row["file_id"]
        if row["score"] < 0.6 and row["top_project"]:
            # only for those we want personalized letters
            j = json.loads((JOB_JSON_DIR/f"{fid}.json").read_text(encoding="utf-8"))
            data[fid] = {
                "job_title": j["title"] if "title" in j else j["job_title"],
                "company":   j.get("company", ""),
                "must_skills": j.get("must_have", []),
                "top_project": row["top_project"],
            }
    return data

def format_skills(skills: list[str]) -> str:
    """Comma-separate and Oxford-comma list."""
    if len(skills) <= 2:
        return " and ".join(skills)
    return ", ".join(skills[:-1]) + " and " + skills[-1]

def generate_cover_letter(meta: dict) -> str:
    prompt = COVER_TMPL.format(
        job_title    = meta["job_title"],
        company      = meta["company"],
        top_project  = meta["top_project"],
        must_skills  = format_skills(meta["must_skills"]),
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

# --- Main ---
def main():
    metas = load_job_metadata()
    if not metas:
        print("🚫 Hiç uygun ilan (score<0.6 ve top_project) bulunamadı.")
        return

    for fid, meta in metas.items():
        letter = generate_cover_letter(meta)
        out_path = OUTPUT_DIR / f"{fid}.md"
        out_path.write_text(letter, encoding="utf-8")
        print(f"✅ {fid} → Cover letter saved ({len(letter.split())} words)")

if __name__ == "__main__":
    main()
