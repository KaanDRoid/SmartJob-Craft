from pathlib import Path
import re, os, json, pandas as pd
from smartjobcraft_day1 import JobParser, SkillNormalizer, score_candidate
from cv_parser import extract_skills_and_languages # Changed import

# parameters
CV_FILE         = Path("Kaan_AK_CV.pdf")     # Path to your CV file
# Patterns for job advertisement files. 
# Supports Spanish (es), Turkish (tr), and English (en) ads in this example.
ADS_PATTERNS    = ["tests/spanish_ad_*.txt", "tests/turkish_ad_*.txt", "tests/english_ad_*.txt"]
USE_OPENAI      = True # Set to False to use a dummy LLM for testing without API calls
CAND_YEARS_EXP  = 3    # Candidate's years of experience
# Keywords to identify relevant educational background for a potential bonus
EDU_KEYWORDS    = {"engineering", "mathematics", "statistics"}

# llms (Language Learning Models)
from smartjobcraft_day1 import _dummy_llm   # Import a dummy LLM for offline testing
if USE_OPENAI:
    from openai import OpenAI
    # Ensure your OPENAI_API_KEY is set as an environment variable
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    def _llm(prompt: str) -> str:
        """Sends a prompt to OpenAI API and gets a JSON response."""
        return client.chat.completions.create(
            model="gpt-4o-mini", # Using a specific OpenAI model
            messages=[{"role":"user","content":prompt}],
            temperature=0, # Low temperature for deterministic output
            response_format={"type":"json_object"}, # Expecting a JSON object as response
        ).choices[0].message.content
else:
    _llm = _dummy_llm # Use the dummy LLM if USE_OPENAI is False

# utils
# Regular expression to split a text file containing multiple job ads.
# It looks for lines starting with '#' followed by 'ilan' (Turkish for 'advertisement') or 'advertisement'.
# Case-insensitive (re.I).
_SPLIT = re.compile(r"^\s*#+\s*(?:i?lan|advertisement)", re.I) 
def split_ads(txt:str):
    """Splits a single text string containing multiple job ads into a list of individual ads."""
    chunk, out = [], []
    for ln in txt.splitlines():
        if _SPLIT.match(ln): # If a separator line is found
            if chunk: out.append("\n".join(chunk)); chunk=[] # Add the collected ad to output
            continue
        chunk.append(ln) # Collect lines of the current ad
    if chunk: out.append("\n".join(chunk)) # Add the last ad
    return [c.strip() for c in out if c.strip()] # Return non-empty, stripped ads

# Regular expression to extract minimum years of experience from a job posting.
# Looks for patterns like "3 years", "5+ yrs", "2 años" (Spanish for years).
_RX_EXP = re.compile(r"(\d+)\s*\+?\s*(?:años|yrs?|years)", re.I)

def min_exp(posting:str)->int:
    """Extracts the minimum years of experience required from a job posting string."""
    m = _RX_EXP.search(posting)
    return int(m.group(1)) if m else 0 # Returns the number of years or 0 if not found

def education_bonus(cv_tokens:set[str])->float:
    """Calculates an education bonus if CV tokens match predefined education keywords."""
    # Returns a small bonus (0.03) if any of the EDU_KEYWORDS are found in the CV tokens (case-insensitive).
    return 0.03 if EDU_KEYWORDS & {t.lower() for t in cv_tokens} else 0.0

# main
def main():
    """Main function to process CV, parse job ads, score them, and save a summary."""
    norm  = SkillNormalizer("tests/skills_variants.csv") # Initialize skill normalizer
    # Extract skills and languages from the CV; we only need skills here.
    cand_skills, _ = extract_skills_and_languages(CV_FILE, norm) 
    parser = JobParser(_llm) # Initialize job parser with the selected LLM

    rows=[] # To store results for each job ad
    print(f"Processing job ads from patterns: {ADS_PATTERNS}")
    for pat in ADS_PATTERNS:
        print(f"Looking for files matching: {pat}")
        found_files = list(Path().glob(pat))
        if not found_files:
            print(f"No files found for pattern: {pat}")
            continue
        for path in found_files:
            print(f"Processing file: {path.name}")
            raw_ad_text = path.read_text(encoding="utf-8")
            for i,ad_text in enumerate(split_ads(raw_ad_text),1):
                print(f"  Parsing ad #{i} from {path.name}...")
                job = parser.parse(ad_text) # Parse the individual job ad
                exp_bonus = 0.0
                needed_exp = min_exp(ad_text) # Get required experience from ad
                if needed_exp:
                    # Calculate experience bonus based on the difference between candidate's experience and required experience.
                    # Max bonus of 0.1, with 0.02 per year of difference.
                    diff = max(0, CAND_YEARS_EXP - needed_exp)
                    exp_bonus = min(0.1, diff*0.02)
                
                edu_bonus = education_bonus(cand_skills) # Calculate education bonus

                # Score the candidate against the job, including experience and education bonuses.
                score = score_candidate(
                    cand_skills, job,
                    exp_bonus=exp_bonus + edu_bonus
                )
                must_have_skills = job.must_have or []
                nice_to_have_skills = job.nice_to_have or []
                # Identify skill gaps by comparing normalized job skills with candidate skills.
                gaps  = set(norm.canonical_set(must_have_skills + nice_to_have_skills)) - cand_skills
                
                file_id = f"{path.stem}__{i}"
                print(f"    Processed ad: {file_id}, Title: {job.title}, Score: {score:.3f}")
                rows.append({
                    "file_id": file_id,
                    "title": job.title,
                    "score": round(score,3),
                    "gaps": ", ".join(sorted(gaps)[:7]) # List up to 7 skill gaps
                })

    if not rows:
        print("No job ads were processed. Please check ADS_PATTERNS and file locations.")
        return

    # Create a DataFrame from the results, sorted by score in descending order.
    df=pd.DataFrame(sorted(rows,key=lambda r:r["score"],reverse=True))
    print("\n--- Job Ad Scoring Summary ---")
    print(df.to_string(index=False))
    
    output_csv_path = Path("summary.csv")
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"\nSummary saved to {output_csv_path}")

if __name__=="__main__":
    main()
