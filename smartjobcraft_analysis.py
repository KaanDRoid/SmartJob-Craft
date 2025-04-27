from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set, Tuple

# Project imports
from smartjobcraft_day1 import (
    JobParser,
    SkillNormalizer,
    score_candidate,
)


# 1. LLM backends 


import json as _json

def _dummy_llm(prompt: str) -> str:
    """Sabit JSON döndürür – hız için."""
    return _json.dumps(
        {
            "job_title": "Stub Title",
            "company": "Stub Co",
            "location": "—",
            "contract_type": None,
            "must_have": ["Python"],
            "nice_to_have": [],
            "visa_requirement": None,
            "language_requirement": [],
        }
    )

USE_OPENAI = True  
if USE_OPENAI:
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _openai_llm(prompt: str) -> str:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    LLM_FN = _openai_llm
else:
    LLM_FN = _dummy_llm


# 2. Config 


ADS_GLOB_PATTERNS = [
    "tests/spanish_ad_*.txt",
    "tests/turkish_ad_*.txt",
    "tests/english_ad_*.txt"
]

OUTPUT_DIR = Path("job_json")
OUTPUT_DIR.mkdir(exist_ok=True)

CANDIDATE_SKILLS: set[str] = {
    "AWS", "Agile", "Airflow", "AutoCAD", "Blender", "C#", "Data Analysis",
    "Docker", "DuckDB", "Excel", "FastAPI", "Feature Engineering", "Git",
    "GitHub", "Godot", "IAM", "Jupyter Notebook", "Kafka", "Kanban",
    "Machine Learning", "Matplotlib", "MediaPipe", "Model Deployment",
    "MongoDB", "NumPy", "OpenCV", "Pandas", "Poetry", "PostgreSQL",
    "PyTorch", "Python", "SQL", "SQLite", "Scikit-learn", "Scrum", "Seaborn",
    "Spark", "TensorFlow", "Terraform", "Unity", "pytest"
}


# regex for splitting ads
SPLIT_REGEX = re.compile(r"^\s*#+\s*i?lan", re.IGNORECASE)  # satır '#ilan ...'

# 3. Helper functions 


def split_ads(text: str) -> List[str]:
    """`#ilan` başlıklarına göre dosyayı ilanlara böler."""
    chunks: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        if SPLIT_REGEX.match(line):
            if current:
                chunks.append("\n".join(current).strip())
                current = []
            continue  # başlık satırını atla
        current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    return [c for c in chunks if c]


def iter_ads(patterns: Iterable[str]):
    for pat in patterns:
        for txt_path in Path().glob(pat):
            raw_file = txt_path.read_text(encoding="utf-8")
            for idx, posting in enumerate(split_ads(raw_file), start=1):
                name = f"{txt_path.stem}__{idx}"
                yield name, posting


def save_json(obj: dict, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# 4. Main 

def main():
    parser = JobParser(LLM_FN)
    normalizer = SkillNormalizer("tests/skills_variants.csv")
    candidate_canon = normalizer.canonical_set(CANDIDATE_SKILLS)

    summary: List[Tuple[str, str, float, List[str]]] = []

    for file_id, raw_posting in iter_ads(ADS_GLOB_PATTERNS):
        job = parser.parse(raw_posting)

        job_must = normalizer.canonical_set(job.must_have or [])
        job_nice = normalizer.canonical_set(job.nice_to_have or [])

        score = score_candidate(candidate_canon, job)
        gaps = list((job_must | job_nice) - candidate_canon)

        summary.append((file_id, job.title, round(score, 3), gaps))

        save_json(job.__dict__, OUTPUT_DIR / f"{file_id}.json")

    # Summary table
    print("\n=== SUMMARY ===")
    import pandas as pd
    df = pd.DataFrame([
        {"file_id": fid, "title": title, "score": sc, "gaps": ", ".join(gaps)}
        for fid, title, sc, gaps in sorted(summary, key=lambda r: r[2], reverse=True)
    ])
    print(df.to_string(index=False))
    df.to_csv("summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
