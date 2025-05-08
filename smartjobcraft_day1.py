"""
SmartJob Craft - Day 1 Implementation
Components:
  • JobParser (LLM-powered, few shot)
  • SkillNormalizer (rapidfuzz + CSV synonyms)
  • Matcher & Scorer (v1.3 formula)
NOTE: LLM calls are stubbed; replace self._call_llm with your engine of choice.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Set

from rapidfuzz import fuzz, process
import openai
import os
# import json # Already imported

# Data Models


@dataclass
class Job:
    title: str
    company: str
    location: str
    contract: str | None
    must_have: List[str]
    nice_to_have: List[str]
    visa_requirement: str | None = None
    language_requirement: List[str] = field(default_factory=list)
    salary_range: str | None = None
    relocation_support: bool | None = None
    remote_option: str | None = None
    industry: str | None = None

@dataclass
class Candidate:
    name: str
    headline: str
    skills: Set[str]
    projects: List[str]
    education: List[str]
    soft_skills: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    years_of_experience: int | None = None


# JobParser – few‑shot prompt template

# This prompt provides examples to the LLM for how to parse job postings.
FEW_SHOT_PROMPT = """SYSTEM: You are an expert HR analyst.
### Example 1 ###
Software Engineer at Google, London. Must have: Python, SQL. Nice to have: React.
### Output ###
{{"job_title":"Software Engineer","company":"Google","location":"London","contract_type":null,"must_have":["Python","SQL"],"nice_to_have":["React"],"visa_requirement":null,"language_requirement":[]}}
### End Example ###
### Example 2 ###
Data Scientist at Amazon, Seattle (remote). Must have: TensorFlow, 3+ yrs exp. Nice to have: AWS, PhD. Visa support available.
### Output ###
{{"job_title":"Data Scientist","company":"Amazon","location":"Seattle","contract_type":"remote","must_have":["TensorFlow","3+ yrs exp"],"nice_to_have":["AWS","PhD"],"visa_requirement":"available","language_requirement":[]}}
### End Example ###
USER: Delimited by ### is a job posting. Return **only** valid JSON with the same keys as above.
###
{job_posting}
###"""

class JobParser:
    """LLM powered job ad parser returning a Job dataclass."""

    def __init__(self, llm_call):
        self._call_llm = llm_call  # func: str -> str (JSON)

    def parse(self, job_posting: str) -> Job:
        prompt = FEW_SHOT_PROMPT.format(job_posting=job_posting.strip())
        llm_response = self._call_llm(prompt)
        # Clean up code block formatting from LLM response
        txt = llm_response.strip()
        if txt.startswith("```"):
            txt = txt.strip("`")
            if txt.lstrip().startswith("json"):
                txt = txt.split("\n", 1)[1]
        try:
            data = json.loads(txt)
            if isinstance(data, list):
                data = data[0] # Take the first element if response is a list
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM response is invalid JSON: {e}\n\nRaw response:\n{txt[:500]}")
        return Job(
            title=data["job_title"],
            company=data["company"],
            location=data["location"],
            contract=data.get("contract_type"), # Use .get for optional fields
            must_have=data["must_have"],
            nice_to_have=data["nice_to_have"],
            visa_requirement=data.get("visa_requirement"),
            language_requirement=data.get("language_requirement", []),
        )

# SkillNormalizer – synonym → canonical mapping

class SkillNormalizer:
    """Normalizes skill names to a canonical form using a CSV of synonyms and fuzzy matching."""
    def __init__(self, csv_path: str | Path = "tests/skills_variants.csv", score_cutoff: int = 90):
        self._canonical: dict[str, str] = {}
        self._score_cutoff = score_cutoff # Threshold for fuzzy matching
        self._load(csv_path)

    def _load(self, path: str | Path):
        """Loads skill synonyms from a CSV file."""
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for canonical, *variants in reader:
                self._canonical[canonical.lower()] = canonical  # Map canonical to itself
                for v in variants:
                    self._canonical[v.lower()] = canonical # Map variants to canonical

    def canonical(self, skill: str) -> str:
        """Returns the canonical form of a skill."""
        key = skill.lower().strip()
        if key in self._canonical:
            return self._canonical[key]
        # Fallback to fuzzy matching if no exact match is found
        match, score, _ = process.extractOne(key, self._canonical.keys(), scorer=fuzz.WRatio)
        if score >= self._score_cutoff:
            return self._canonical[match]
        return skill  # Return as‑is if no good match

    def canonical_set(self, skills: Iterable[str]) -> Set[str]:
        """Returns a set of canonical skill names from an iterable of skills."""
        return {self.canonical(s) for s in skills}


# Matcher & Scorer


def calculate_exp_weight(years_of_experience: int | None, job_min_exp: int = 0) -> float:
    """Calculates a weight based on years of experience compared to job minimum.
       Returns a small bonus if candidate experience meets or exceeds job requirements.
    """
    if years_of_experience is None:
        return 0.0
    if years_of_experience >= job_min_exp:
        # Bonus increases with more experience, capped at 0.1
        return min(0.1, (years_of_experience - job_min_exp) * 0.02)
    return 0.0 # No bonus if experience is less than required

def score_candidate(
    candidate_skills: Set[str],
    job: Job,
    job_freq: Counter[str] | None = None, # Optional: frequency of skills in job posting for weighting
    exp_bonus: float = 0.0, # Bonus for experience match
    project_bonus: float = 0.0, # Bonus for relevant project experience
) -> float:
    """Scores a candidate against a job based on skill match, experience, and projects."""
    job_must = job.must_have or []
    job_nice = job.nice_to_have or []
    # Default frequency: each skill counts as 1 if not provided
    job_freq = job_freq or Counter({s: 1 for s in job_must + job_nice})
    
    must_have = set(job_must)
    nice_to_have = set(job_nice)
    
    # Skills the candidate has that are mentioned in the job posting
    matches = candidate_skills & (must_have | nice_to_have)

    # Score for matching must-have skills, weighted by frequency
    mh_score = sum(job_freq.get(s, 1) for s in matches & must_have) / max(1, sum(job_freq.values()))
    # Score for matching nice-to-have skills
    nt_score = len(matches & nice_to_have) / max(1, len(nice_to_have))
    
    # Base score combines must-have and nice-to-have matches
    # Must-have skills are weighted more heavily (70% vs 30%)
    base_score = 0.7 * mh_score + 0.3 * nt_score
    
    # Final score includes bonuses for experience and projects
    return base_score + exp_bonus + project_bonus


# Stub LLM call (replace in production)


def _dummy_llm(prompt: str) -> str:
    """Return a hard‑coded minimal JSON so unit tests do not hit an API.
       Useful for offline testing or when API access is unavailable.
    """
    _template = {
        "job_title": "Stub Title",
        "company": "Stub Co",
        "location": "—", # Placeholder for location
        "contract_type": None,
        "must_have": ["Python"],
        "nice_to_have": [],
        "visa_requirement": None,
        "language_requirement": [],
    }
    return json.dumps(_template)

def _openai_llm(prompt: str) -> str:
    """Calls the OpenAI API to parse the job posting.
       Ensure the OPENAI_API_KEY environment variable is set.
    """
    # IMPORTANT: Always use environment variables for API keys.
    # Do not hardcode API keys in your source code.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        # Fallback to dummy_llm or raise an error if API key is crucial
        return _dummy_llm(prompt) # Or raise Exception("API Key not found")
        
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # A capable and cost-effective model
            messages=[{"role": "user", "content": prompt}],
            temperature=0 # Low temperature for deterministic output
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        # Fallback or error handling
        return _dummy_llm(prompt) # Example: fallback to dummy on error

# Quick self‑test when run as script

if __name__ == "__main__":
    print("Running self-test for JobParser and SkillNormalizer...")
    # 1) Initialize Parser & Normalizer
    # Use _openai_llm for actual API calls, or _dummy_llm for testing without API access.
    parser = JobParser(_openai_llm) # Using OpenAI LLM by default for self-test
    # Path to the skill variants CSV, relative to this script's location
    skills_csv_path = Path(__file__).parent / "tests/skills_variants.csv"
    normalizer = SkillNormalizer(skills_csv_path)

    # 2) Iterate through test job ad files
    test_ads_dir = Path("tests") # Assuming test files are in a 'tests' subdirectory

    print("\n--- Processing Spanish Ads ---")
    for txt_file in test_ads_dir.glob("spanish_ad_*.txt"):
        raw_posting = txt_file.read_text(encoding="utf-8")
        job = parser.parse(raw_posting)

        # Get canonical skill names
        canon_must  = normalizer.canonical_set(job.must_have)
        canon_nice  = normalizer.canonical_set(job.nice_to_have)

        print(f"---- File: {txt_file.name} ----")
        print(f"Title: {job.title} | Company: {job.company}")
        print(f"Must-have Skills: {canon_must}")
        print(f"Nice-to-have Skills: {canon_nice}")
        print()

    print("\n--- Processing Turkish Ads ---")
    for txt_file in test_ads_dir.glob("turkish_ad_*.txt"):
        raw_posting = txt_file.read_text(encoding="utf-8")
        job = parser.parse(raw_posting)
        canon_must = normalizer.canonical_set(job.must_have)
        print(f"---- File: {txt_file.name} ----")
        print(f"Must-have Skills: {canon_must}")
        print()

    print("\n--- Processing English Ads ---")
    for txt_file in test_ads_dir.glob("english_ad_*.txt"):
        raw_posting = txt_file.read_text(encoding="utf-8")
        job = parser.parse(raw_posting)
        canon_must = normalizer.canonical_set(job.must_have)
        print(f"---- File: {txt_file.name} ----")
        print(f"Must-have Skills: {canon_must}")
        print()
    print("Self-test finished.")