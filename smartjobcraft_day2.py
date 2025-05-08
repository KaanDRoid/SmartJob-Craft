# smartjobcraft_day2.py
"""
SmartJob Craft - Day 2 Implementation
This module extends the base functionality with:
  • Project ranking based on job requirements
  • Skill gap analysis and learning plan generation
  • Enhanced candidate scoring with education and experience bonuses
"""

from __future__ import annotations
import os, json, re
from pathlib import Path
from typing import List, Tuple, Set

from smartjobcraft_day1 import JobParser, SkillNormalizer, Candidate, score_candidate, _dummy_llm, _openai_llm
from cv_parser      import extract_skills_and_languages
from smartjobcraft_gaprec import LLM as GAP_LLM, PROMPT as GAP_PROMPT, THRESHOLD as GAP_THRESHOLD 

# --- ProjectRanker (Day 2) ---
# Regular expression to identify meaningful tokens in text (words, numbers, and some special characters)
_RX_TOKEN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\+\#\.\-]{2,}")

def _tokens(text: str) -> List[str]:
    """Extract meaningful tokens from text using regex pattern."""
    return _RX_TOKEN.findall(text)

def rank_projects(
    job, cand_projects: List[str], normalizer: SkillNormalizer
) -> Tuple[str, float]:
    """
    Rank candidate's projects based on their relevance to job requirements.
    
    Args:
        job: Job object containing must-have and nice-to-have skills
        cand_projects: List of candidate's project descriptions
        normalizer: SkillNormalizer to standardize skill names
    
    Returns:
        Tuple containing the best matching project and its coverage score
    """
    must = {normalizer.canonical(s).lower() for s in job.must_have or []}
    nice = {normalizer.canonical(s).lower() for s in job.nice_to_have or []}
    top_proj, best_cov = "", 0.0
    for proj in cand_projects:
        toks = [t.lower() for t in _tokens(proj)]
        proj_skills = {normalizer.canonical(t).lower() for t in toks}
        cov = (len(must & proj_skills) + 0.5 * len(nice & proj_skills)) \
              / (len(must) + 0.5 * len(nice) or 1)
        if cov > best_cov:
            best_cov, top_proj = cov, proj
    return top_proj, best_cov

def compute_project_bonus(coverage: float) -> float:
    """
    Calculate score bonus based on project coverage.
    
    Args:
        coverage: Project coverage score between 0 and 1
    
    Returns:
        Bonus score value
    """
    return 0.05 if coverage > 0 else 0.0

# --- Helpers ---
# Regular expression to find years of experience requirements in job postings
_RX_EXP = re.compile(r"(\d+)\s*\+?\s*(?:años|yrs?|years)", re.I)

def min_exp(posting: str) -> int:
    """
    Extract minimum years of experience required from job posting.
    Searches for patterns like "3+ years", "5 años", etc.
    
    Args:
        posting: Raw job posting text
        
    Returns:
        Integer representing years of experience, or 0 if not found
    """
    m = _RX_EXP.search(posting)
    return int(m.group(1)) if m else 0

# Keywords indicating relevant educational background
EDU_KEYWORDS = {"engineering","mathematics","statistics"}

def education_bonus(cand_skills: Set[str]) -> float:
    """
    Calculate education bonus based on candidate's skills.
    
    Args:
        cand_skills: Set of candidate's skills
        
    Returns:
        Education bonus score if candidate has relevant educational keywords
    """
    return 0.03 if EDU_KEYWORDS & {s.lower() for s in cand_skills} else 0.0

def split_ads(text: str) -> List[str]:
    """
    Split a text file containing multiple job ads into individual ads.
    
    Args:
        text: Raw text containing multiple job postings
        
    Returns:
        List of individual job posting texts
    """
    regex = re.compile(r"^\s*#+\s*i?lan", re.I)  # Matches section headers like "# ilan" or "### ILAN"
    chunks, cur = [], []
    for ln in text.splitlines():
        if regex.match(ln):
            if cur: chunks.append("\n".join(cur).strip()); cur=[]
            continue
        cur.append(ln)
    if cur: chunks.append("\n".join(cur).strip())
    return [c for c in chunks if c]

def read_txt_paths(globs) -> List[Path]:
    """
    Read file paths matching the provided glob patterns.
    
    Args:
        globs: List of glob patterns to match files
        
    Returns:
        List of Path objects for matching files
    """
    p=[]
    for g in globs: p += list(Path().glob(g))
    return p

# --- Main ---
def main():
    """
    Main function that:
    1. Loads and processes a candidate's CV
    2. Extracts projects from the CV
    3. Processes job ads and scores the candidate against each job
    4. Generates learning plans for skill gaps
    5. Saves results to files
    """
    # 1) Load CV and extract skills
    norm   = SkillNormalizer("tests/skills_variants.csv")
    cand_sk, _ = extract_skills_and_languages(Path("Kaan_AK_CV.pdf"), norm)
    
    # 2) Extract projects from CV
    # Import PyPDF2 for PDF extraction
    import PyPDF2
    
    def extract_projects(cv_path: str) -> list[str]:
        """
        Extract project descriptions from a PDF CV.
        
        Args:
            cv_path: Path to the PDF CV file
            
        Returns:
            List of project description texts
        """
        # Read PDF text
        with open(cv_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted: # Check if text extraction was successful
                    text += extracted + "\n"

        print("--- Text Extracted from PDF ---")
        # print(text) # Keep this commented unless needed for deep debugging

        projects_text = []
        current_project_lines = []
        in_section = False
        found_first_project = False
        lines = text.splitlines()

        # Regex to identify the start of the Projects section
        section_start_regex = re.compile(r"^\s*Projects\s*$", re.IGNORECASE)
        
        # Improved regex to identify ONLY main project titles, not sub-headers
        main_project_regex = re.compile(
            r"^(E-commerce|LinguaLink|Big Data).*?:",
            re.IGNORECASE
        )
        
        # Regex to identify potential section end markers
        section_end_regex = re.compile(r"^\s*Technical Skills\s*$", re.IGNORECASE)

        for i, line in enumerate(lines):
            line_strip = line.strip()
            
            # 1. Find the Projects section
            if not in_section:
                if "Projects" in line_strip:
                    in_section = True
                continue  # Skip until we find the Projects section

            # 2. We're in the Projects section
            # Check if this line starts a new main project
            is_main_project = bool(main_project_regex.match(line))
            is_section_end = bool(section_end_regex.match(line_strip))
            
            if is_main_project or is_section_end:
                # If we already found a project, save the current one before starting the next
                if current_project_lines:
                    project_text = "\n".join(current_project_lines).strip()
                    if project_text:  # Only add non-empty projects
                        projects_text.append(project_text)
                    # Reset for the next project
                    current_project_lines = []
                
                if is_section_end:
                    break  # End of the Projects section
                
                # Start collecting the new project
                if is_main_project:
                    current_project_lines.append(line)
                    found_first_project = True
            
            # Add lines to the current project if we're already tracking one
            elif found_first_project:
                current_project_lines.append(line)

        # Don't forget to add the last project if there is one
        if current_project_lines:
            project_text = "\n".join(current_project_lines).strip()
            if project_text:
                projects_text.append(project_text)

        if not projects_text:
            print("DEBUG: No project text blocks extracted.")
        else:
            print(f"DEBUG: Extracted {len(projects_text)} project text blocks.")

        return projects_text
    
    cand_projs = extract_projects("Kaan_AK_CV.pdf")

    # 3) Initialize job parser
    llm_fn = (_dummy_llm if os.getenv("USE_DUMMY") else _openai_llm)
    parser = JobParser(llm_fn)

    # Create output directories if they don't exist
    OUTPUT_JSON = Path("job_json2"); OUTPUT_JSON.mkdir(exist_ok=True)
    LEARN_DIR   = Path("learning_plans"); LEARN_DIR.mkdir(exist_ok=True)

    # Process all job ads and evaluate candidate
    rows = []
    for path in read_txt_paths(["tests/spanish_ad_*.txt","tests/turkish_ad_*.txt"]):
        raw = path.read_text(encoding="utf-8")
        for idx, ad in enumerate(split_ads(raw),1):
            fid = f"{path.stem}__{idx}"
            job = parser.parse(ad)

            # Normalize skills for comparison
            job_must = norm.canonical_set(job.must_have or [])
            job_nice = norm.canonical_set(job.nice_to_have or [])

            # Calculate experience and education bonuses
            exp_need = min_exp(ad)
            years = 3  # Candidate's years of experience (could be extracted from CV)
            exp_bonus = max(0, min(0.1, (years-exp_need)*0.02))
            edu_bonus = education_bonus(cand_sk)

            # Calculate project relevance bonus
            top_proj, cov = rank_projects(job, cand_projs, norm)
            proj_bonus = compute_project_bonus(cov)

            # Calculate final score with all bonuses
            score = score_candidate(cand_sk, job, exp_bonus=exp_bonus+edu_bonus, project_bonus=proj_bonus)

            # Identify skill gaps
            gaps = list((job_must|job_nice) - cand_sk)
            rows.append({"file_id":fid,"title":job.title,"score":round(score,3),"gaps":", ".join(gaps),"top_project":top_proj})

            # Save job details as JSON
            OUTPUT_JSON.joinpath(f"{fid}.json").write_text(
                json.dumps(job.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # Generate learning plan for significant skill gaps
            if score < GAP_THRESHOLD and gaps:
                prompt = GAP_PROMPT.format(title=job.title, missing=", ".join(gaps))
                plan = json.loads(GAP_LLM(prompt))["plan"]
                LEARN_DIR.joinpath(f"{fid}.md").write_text(plan, encoding="utf-8")
                print(f"✅ {fid} → learning plan")

    # Write summary to CSV and display results
    import pandas as pd
    df = pd.DataFrame(sorted(rows, key=lambda r: r["score"], reverse=True))
    df.to_csv("summary2.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))

if __name__=="__main__":
    main()
