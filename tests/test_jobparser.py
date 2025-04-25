import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


from smartjobcraft_day1 import JobParser, SkillNormalizer, score_candidate
import json

def dummy_llm(_):
    return json.dumps({
        "job_title": "Data Engineer",
        "company": "Hiberus",
        "location": "Barcelona",
        "contract_type": "Full-time",
        "must_have": ["Tableau", "SQL"],
        "nice_to_have": ["AWS"],
        "visa_requirement": None,
        "language_requirement": ["English"]
    })

def test_parse_and_score():
    parser = JobParser(dummy_llm)
    job = parser.parse("fake ad")

    
    csv_path = Path(__file__).resolve().parent / "skills_variants.csv"
    normalizer = SkillNormalizer(csv_path)

    cand = normalizer.canonical_set({"sql", "aws"})
    assert score_candidate(cand, job) > 0
