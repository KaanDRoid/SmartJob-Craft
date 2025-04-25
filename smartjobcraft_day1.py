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
import json

# Data Models


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


FEW_SHOT_PROMPT = """SYSTEM: You are an expert HR analyst.\n### Example\u00a01 ###\nSoftware Engineer at Google, London. Must have: Python, SQL. Nice to have: React.\n### Output ###\n{{\"job_title\":\"Software Engineer\",\"company\":\"Google\",\"location\":\"London\",\"contract_type\":null,\"must_have\":[\"Python\",\"SQL\"],\"nice_to_have\":[\"React\"],\"visa_requirement\":null,\"language_requirement\":[]}}\n### End Example ###\n### Example\u00a02 ###\nData Scientist at Amazon, Seattle (remote). Must have: TensorFlow, 3+ yrs exp. Nice to have: AWS, PhD. Visa support available.\n### Output ###\n{{\"job_title\":\"Data Scientist\",\"company\":\"Amazon\",\"location\":\"Seattle\",\"contract_type\":\"remote\",\"must_have\":[\"TensorFlow\",\"3+ yrs exp\"],\"nice_to_have\":[\"AWS\",\"PhD\"],\"visa_requirement\":\"available\",\"language_requirement\":[]}}\n### End Example ###\nUSER: Delimited by ### is a job posting. Return **only** valid JSON with the same keys as above.\n###\n{job_posting}\n###"""

class JobParser:
    """LLM powered job ad parser returning a Job dataclass."""

    def __init__(self, llm_call):
        self._call_llm = llm_call  # func: str -> str (JSON)

    def parse(self, job_posting: str) -> Job:
        prompt = FEW_SHOT_PROMPT.format(job_posting=job_posting.strip())
        llm_response = self._call_llm(prompt)

        # --- TEMİZLE
        txt = llm_response.strip()
        if txt.startswith("```"):
            txt = txt.strip("`")
            # ilk satır yalnızca "json" ise at 
            if txt.lower().startswith("json"):
                txt = txt.split("\n", 1)[1]
        # HA BURAYA KADAR UIYYY

        data = json.loads(txt)
        if isinstance(data, list):
            data = data[0]
        return Job(
            title=data["job_title"],
            company=data["company"],
            location=data["location"],
            contract=data.get("contract_type"),
            must_have=data["must_have"],
            nice_to_have=data["nice_to_have"],
            visa_requirement=data.get("visa_requirement"),
            language_requirement=data.get("language_requirement", []),
        )

# SkillNormalizer – synonym → canonical mapping

class SkillNormalizer:
    def __init__(self, csv_path: str | Path = "tests/skills_variants.csv", score_cutoff: int = 90):
        self._canonical: dict[str, str] = {}
        self._score_cutoff = score_cutoff
        self._load(csv_path)

    def _load(self, path: str | Path):
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row or not row[0].strip():
                    continue  # boş satır veya ilk hücresi boşsa atla
                canonical, *variants = row
                self._canonical[canonical.lower()] = canonical  # keep self map
                for v in variants:
                    self._canonical[v.lower()] = canonical

    def canonical(self, skill: str) -> str:
        key = skill.lower().strip()
        if key in self._canonical:
            return self._canonical[key]
        # fuzzy match fallback
        match, score, _ = process.extractOne(key, self._canonical.keys(), scorer=fuzz.WRatio)
        if score >= self._score_cutoff:
            return self._canonical[match]
        return skill  # return as‑is if no good match

    def canonical_set(self, skills: Iterable[str]) -> Set[str]:
        return {self.canonical(s) for s in skills}


# Matcher & Scorer
def calculate_exp_weight(years_of_experience: int | None, job_min_exp: int = 0) -> float:
    if years_of_experience is None:
        return 0.0
    if years_of_experience >= job_min_exp:
        return min(0.1, (years_of_experience - job_min_exp) * 0.02)
    return 0.0

def score_candidate(
    candidate_skills: Set[str],
    job: Job,
    job_freq: Counter[str] | None = None,
    exp_bonus: float = 0.0,
    project_bonus: float = 0.0,
) -> float:
    # None gelebilecek alanları güvenli listeye çevirecek umarım
    job_must = job.must_have or []
    job_nice = job.nice_to_have or []
    job_freq = job_freq or Counter({s: 1 for s in job_must + job_nice})
    must_have = set(job_must)
    nice_to_have = set(job_nice)
    matches = candidate_skills & (must_have | nice_to_have)

    mh_score = sum(job_freq.get(s, 1) for s in matches & must_have) / max(1, sum(job_freq.values()))
    nt_score = len(matches & nice_to_have) / max(1, len(nice_to_have))
    base_score = 0.7 * mh_score + 0.3 * nt_score
    return base_score + exp_bonus + project_bonus


# Stub LLM call 

USE_OPENAI = True

def _openai_llm(prompt: str) -> str:
    import openai
    import os
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

# Quick self‑test when run as script

if __name__ == "__main__":
    # 1) Parser & Normalizer
    parser = JobParser(_openai_llm) if USE_OPENAI else JobParser(_dummy_llm)
    normalizer = SkillNormalizer(Path(__file__).parent / "tests/skills_variants.csv")

    # 2) Tek tek ilan dosyalarını gez
    for txt in Path("tests").glob("spanish_ad_*.txt"):
        raw = txt.read_text(encoding="utf-8")
        job = parser.parse(raw)

        # Canonical beceriler
        canon_must  = normalizer.canonical_set(job.must_have)
        canon_nice  = normalizer.canonical_set(job.nice_to_have)

        print("----", txt.name)
        print("Title :", job.title, "| Company:", job.company)
        print("Must  :", canon_must)
        print("Nice  :", canon_nice)
        print()

    # Türkçe ilanlar için benzer döngü
    for txt in Path("tests").glob("turkish_ad_*.txt"):
        raw = txt.read_text(encoding="utf-8")
        job = parser.parse(raw)
        canon_must = normalizer.canonical_set(job.must_have)
        print("----", txt.name)
        print("Must  :", canon_must)