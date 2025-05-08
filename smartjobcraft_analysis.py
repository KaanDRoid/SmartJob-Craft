"""smartjobcraft_analysis.py

Batch-analysis helper script
============================
- Reads: *tests/spanish_ad_*.txt*, *tests/turkish_ad_*.txt* (each file can contain multiple #job sections)
- Uses: smartjobcraft_day1.JobParser, SkillNormalizer, score_candidate
- Outputs: JSON for each job posting -> job_json/<file>__<n>.json
- To console: score table + list of missing skills

How to run
----------
$ python smartjobcraft_analysis.py

LLM selection
-------------
Default: `_dummy_llm` (fixed JSON). To use a real model, set `USE_OPENAI = True`,
and set the OPENAI_API_KEY environment variable.
"""
from __future__ import annotations

import json
import re
from collections import Counter # Counter is not used in this script, consider removing if not needed elsewhere.
from pathlib import Path
from typing import Iterable, List, Set, Tuple # Set and Counter are not used, consider removing.

# Project imports
from smartjobcraft_day1 import (
    JobParser,
    SkillNormalizer,
    score_candidate,
)


# 1. LLM backends


# import json as _json # json is already imported as json, _json alias is not necessary.

def _dummy_llm(prompt: str) -> str:
    """Returns a fixed JSON string for speed and offline testing."""
    return json.dumps(
        {
            "job_title": "Stub Title",
            "company": "Stub Co",
            "location": "—", # Placeholder for location
            "contract_type": None,
            "must_have": ["Python"],
            "nice_to_have": [],
            "visa_requirement": None,
            "language_requirement": [],
        }
    )

USE_OPENAI = True # Set to False to use the dummy LLM for testing without API calls.
if USE_OPENAI:
    import os
    from openai import OpenAI

    # It's good practice to initialize the client once if the API key is available.
    # This avoids re-initializing on every call to _openai_llm.
    # Ensure OPENAI_API_KEY is set in your environment variables.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. LLM calls will use the dummy implementation.")
        # Fallback to dummy if API key is not set
        LLM_FN = _dummy_llm
    else:
        client = OpenAI(api_key=api_key)
        def _openai_llm(prompt: str) -> str:
            """Makes a call to the OpenAI API to parse the job posting."""
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini", # Using a cost-effective and capable model
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0, # Low temperature for more deterministic, less creative output
                    response_format={"type": "json_object"}, # Ensures the response is JSON formatted
                )
                return resp.choices[0].message.content
            except Exception as e:
                print(f"Error during OpenAI API call: {e}. Falling back to dummy LLM.")
                return _dummy_llm(prompt) # Fallback to dummy on API error
        LLM_FN = _openai_llm
else:
    LLM_FN = _dummy_llm


# 2. Config

# Glob patterns to find job advertisement text files
ADS_GLOB_PATTERNS = [
    "tests/spanish_ad_*.txt",
    "tests/turkish_ad_*.txt",
    "tests/english_ad_*.txt"
]

# Directory to save the parsed JSON output for each job ad
OUTPUT_DIR = Path("job_json")
OUTPUT_DIR.mkdir(exist_ok=True) # Create the directory if it doesn't exist

# Define the candidate's skills. This set will be used to match against job requirements.
# It's important to keep this list updated with the candidate's actual skills.
CANDIDATE_SKILLS: set[str] = {
    "AWS", "Agile", "Airflow", "AutoCAD", "Blender", "C#", "Data Analysis",
    "Docker", "DuckDB", "Excel", "FastAPI", "Feature Engineering", "Git",
    "GitHub", "Godot", "IAM", "Jupyter Notebook", "Kafka", "Kanban",
    "Machine Learning", "Matplotlib", "MediaPipe", "Model Deployment",
    "MongoDB", "NumPy", "OpenCV", "Pandas", "Poetry", "PostgreSQL",
    "PyTorch", "Python", "SQL", "SQLite", "Scikit-learn", "Scrum", "Seaborn",
    "Spark", "TensorFlow", "Terraform", "Unity", "pytest"
}


# Regex for splitting ads within a single file.
# It looks for lines starting with one or more '#' characters, optionally followed by 'ilan'.
SPLIT_REGEX = re.compile(r"^\s*#+\s*i?lan", re.IGNORECASE) # Line starts with '#ilan ...'

# 3. Helper functions


def split_ads(text: str) -> List[str]:
    """Splits a text file into multiple job ads based on `#ilan` headings."""
    chunks: List[str] = []
    current_chunk_lines: List[str] = []
    for line in text.splitlines():
        if SPLIT_REGEX.match(line):
            if current_chunk_lines: # If there are lines in the current chunk, save it
                chunks.append("\n".join(current_chunk_lines).strip())
                current_chunk_lines = [] # Reset for the next chunk
            continue  # Skip the header line itself
        current_chunk_lines.append(line)
    if current_chunk_lines: # Add the last chunk if any
        chunks.append("\n".join(current_chunk_lines).strip())
    return [c for c in chunks if c] # Return only non-empty chunks


def iter_ads(patterns: Iterable[str]):
    """Iterates over job ad files matching the given glob patterns and yields individual ads."""
    for pat in patterns:
        for txt_path in Path().glob(pat):
            raw_file_content = txt_path.read_text(encoding="utf-8")
            for idx, posting_text in enumerate(split_ads(raw_file_content), start=1):
                # Create a unique ID for each job ad based on filename and index
                name = f"{txt_path.stem}__{idx}"
                yield name, posting_text


def save_json(obj: dict, path: Path):
    """Saves a dictionary object to a JSON file with UTF-8 encoding and indentation."""
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# 4. Main

def main():
    """Main function to parse job ads, score them against candidate skills, and save results."""
    # Initialize the JobParser with the selected LLM function (OpenAI or dummy)
    parser = JobParser(LLM_FN)
    # Initialize the SkillNormalizer with the path to the skill variants CSV file
    # This file helps in mapping different skill names (e.g., "JS", "Javascript") to a canonical form.
    normalizer = SkillNormalizer("tests/skills_variants.csv")
    # Get the canonical (standardized) set of the candidate's skills
    candidate_canon_skills = normalizer.canonical_set(CANDIDATE_SKILLS)

    # List to store summary data for each processed job ad
    summary_data: List[Tuple[str, str, float, List[str]]] = []

    print("Processing job advertisements...")
    for file_id, raw_posting_text in iter_ads(ADS_GLOB_PATTERNS):
        print(f"Parsing: {file_id}")
        # Parse the raw job posting text to extract structured information (Job object)
        job = parser.parse(raw_posting_text)

        # Normalize the must-have and nice-to-have skills from the job posting
        job_must_have_canon = normalizer.canonical_set(job.must_have or [])
        job_nice_to_have_canon = normalizer.canonical_set(job.nice_to_have or [])

        # Score the candidate against the job based on skill match
        # The score_candidate function is defined in smartjobcraft_day1.py
        score = score_candidate(candidate_canon_skills, job)
        # Identify skill gaps: skills required by the job but not present in the candidate's skill set
        gaps = list((job_must_have_canon | job_nice_to_have_canon) - candidate_canon_skills)

        # Append the results to the summary list
        summary_data.append((file_id, job.title, round(score, 3), gaps))

        # Save the parsed job data (Job object) as a JSON file
        save_json(job.__dict__, OUTPUT_DIR / f"{file_id}.json")

    # Generate and print a summary table of the results
    print("\n=== SUMMARY OF JOB MATCHING RESULTS ===")
    import pandas as pd # Using pandas for easy table formatting and CSV export
    # Create a DataFrame from the summary data, sorting by score in descending order
    summary_df = pd.DataFrame([
        {"file_id": fid, "title": title, "score": sc, "gaps": ", ".join(sorted(gaps))}
        for fid, title, sc, gaps in sorted(summary_data, key=lambda r: r[2], reverse=True)
    ])
    print(summary_df.to_string(index=False)) # Print the table to the console
    # Save the summary table to a CSV file
    summary_df.to_csv("summary.csv", index=False, encoding="utf-8-sig") # utf-8-sig for Excel compatibility
    print("\nProcessing complete. Summary saved to summary.csv and individual JSONs to job_json/ directory.")

if __name__ == "__main__":
    main()
