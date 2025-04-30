"""
project_ranker.py
-----------------
Scores candidate CV projects against a job’s must-have skills.

Functions:
  - rank_projects(candidate_projects, must_have_skills):
      returns a list of (project, relevance_ratio) sorted desc.
  - compute_project_bonus(relevance_ratio, max_bonus=0.05):
      maps top relevance to a small bonus for scoring.

Usage:
  from project_ranker import rank_projects, compute_project_bonus

Integration:
  ranked = rank_projects(cand_projects, job.must_have)
  top_proj, top_ratio = ranked[0]
  bonus = compute_project_bonus(top_ratio)
  score = score_candidate(..., project_bonus=bonus)
"""

import re
from typing import List, Tuple
from smartjobcraft_day1 import SkillNormalizer

# Precompile regex for tokenizing project descriptions
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\+\#\.\-]{2,}")


def rank_projects(
    candidate_projects: List[str],
    must_have_skills: List[str]
) -> List[Tuple[str, float]]:
    """
    Rank each project by how well it covers the must-have skills.

    - Normalizes skills via SkillNormalizer.
    - For each project description/title, counts normalized skill mentions.
    - Returns list of (project, ratio) sorted descending by ratio.

    ratio = matched_skills / total_must_have_skills
    """
    # Load normalizer and canonicalize must-have skills
    normalizer = SkillNormalizer()
    canonical_skills = {normalizer.canonical(s) for s in must_have_skills}
    total = len(canonical_skills) or 1

    scored: List[Tuple[str, float]] = []
    for proj in candidate_projects:
        text = proj.lower()
        # count exact occurrences of each canonical skill in project text
        matches = sum(1 for skill in canonical_skills if skill.lower() in text)
        ratio = matches / total
        scored.append((proj, ratio))

    # Sort by descending ratio
    return sorted(scored, key=lambda x: x[1], reverse=True)


def compute_project_bonus(
    relevance_ratio: float,
    max_bonus: float = 0.05
) -> float:
    """
    Compute a small bonus proportional to project relevance.
    If the top project covers all skills (ratio=1.0), returns max_bonus.
    """
    return max_bonus * max(0.0, min(1.0, relevance_ratio))
