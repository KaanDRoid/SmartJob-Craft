"""
project_ranker.py
-----------------
This module is responsible for evaluating and ranking a candidate's CV projects 
against the essential skills required for a specific job.

Key Functions:
  - rank_projects(candidate_projects: List[str], must_have_skills: List[str]) -> List[Tuple[str, float]]:
      Takes a list of project descriptions from a candidate's CV and a list of 
      must-have skills for a job. It returns a list of tuples, where each tuple 
      contains a project and its relevance ratio. The list is sorted in descending 
      order based on this ratio.
      The relevance ratio is calculated as: (number of matched must-have skills in project) / (total number of must-have skills).

  - compute_project_bonus(relevance_ratio: float, max_bonus: float = 0.05) -> float:
      Calculates a small bonus score based on the relevance ratio of the top-ranked project.
      This bonus can be added to the candidate's overall job match score.
      If a project perfectly covers all must-have skills (ratio = 1.0), the function returns the `max_bonus`.

Usage Example:
  from project_ranker import rank_projects, compute_project_bonus
  from smartjobcraft_day1 import Job, Candidate # Assuming Job and Candidate dataclasses are defined

  # Assume 'candidate_cv_projects' is a list of project strings from the CV
  # Assume 'job_requirements' is a Job object with a 'must_have' list of skills
  ranked_cv_projects = rank_projects(candidate_cv_projects, job_requirements.must_have)
  
  if ranked_cv_projects:
      top_project_description, top_project_ratio = ranked_cv_projects[0]
      project_relevance_bonus = compute_project_bonus(top_project_ratio)
      # This bonus can then be used in the overall candidate scoring logic
      # final_score = score_candidate_function(..., project_bonus=project_relevance_bonus)

Integration Notes:
  This module integrates with other parts of the SmartJob Craft system:
  - It uses `SkillNormalizer` (from `smartjobcraft_day1`) to ensure that skills are compared 
    in a consistent, canonical form (e.g., "Python" and "python programming" are treated as the same skill).
  - The output (project bonus) is intended to be used by a master scoring function that 
    evaluates the overall fit of a candidate for a job.
"""

import re
from typing import List, Tuple
from smartjobcraft_day1 import SkillNormalizer # Used for standardizing skill names

# Pre-compile a regular expression for tokenizing project descriptions.
# This helps in breaking down project text into words or skill-like terms.
# It matches sequences of alphanumeric characters, including common technical symbols like '+', '#', '.', '-'.
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\+\#\.\-]{2,}")


def rank_projects(
    candidate_projects: List[str], 
    must_have_skills: List[str]
) -> List[Tuple[str, float]]:
    """
    Ranks candidate projects based on their coverage of must-have job skills.

    Args:
        candidate_projects: A list of strings, where each string is a description of a candidate's project.
        must_have_skills: A list of strings, representing the essential skills required for the job.

    Returns:
        A list of tuples, each containing the project description (string) and its 
        relevance ratio (float). The list is sorted in descending order by the ratio.
        The ratio is calculated as: (matched skills in project) / (total must-have skills).
    """
    # Initialize the SkillNormalizer. This utility converts different variations of skill names
    # (e.g., "JS", "JavaScript") into a single, canonical form for accurate comparison.
    normalizer = SkillNormalizer() 
    
    # Convert the list of must-have skills from the job description into a set of their
    # canonical forms. Using a set allows for efficient checking of skill presence.
    canonical_must_skills = {normalizer.canonical(s) for s in must_have_skills}
    
    # Determine the total number of unique, canonical must-have skills.
    # This will be the denominator in our relevance ratio. If there are no must-have skills,
    # default to 1 to avoid division by zero.
    total_must_skills = len(canonical_must_skills) or 1

    scored_projects: List[Tuple[str, float]] = []
    for project_description in candidate_projects:
        # Convert the project description to lowercase to ensure case-insensitive matching.
        project_text_lower = project_description.lower()
        
        # Count how many of the canonical must-have skills are mentioned in the project description.
        # This is a simple substring check.
        # For more advanced matching, one could tokenize the project_text_lower and compare tokens.
        num_matched_skills = sum(1 for skill in canonical_must_skills if skill.lower() in project_text_lower)
        
        # Calculate the relevance ratio for the current project.
        relevance_ratio = num_matched_skills / total_must_skills
        scored_projects.append((project_description, relevance_ratio))

    # Sort the projects by their relevance ratio in descending order.
    # Projects that cover more of the must-have skills will appear first.
    return sorted(scored_projects, key=lambda x: x[1], reverse=True)


def compute_project_bonus(
    relevance_ratio: float, 
    max_bonus: float = 0.05
) -> float:
    """
    Computes a bonus score based on the project's relevance ratio.
    The bonus is proportional to the relevance ratio, capped by `max_bonus`.

    Args:
        relevance_ratio: The relevance ratio of a project (typically the top-ranked one),
                         ranging from 0.0 to 1.0.
        max_bonus: The maximum possible bonus to award (default is 0.05, or 5%).

    Returns:
        A float representing the calculated bonus. This value will be between 0.0 and `max_bonus`.
    """
    # Ensure the relevance_ratio is within the expected [0.0, 1.0] range before calculating the bonus.
    # max(0.0, ...) handles cases where ratio might be negative (though unlikely here).
    # min(1.0, ...) handles cases where ratio might exceed 1.0 (also unlikely if calculated correctly).
    normalized_ratio = max(0.0, min(1.0, relevance_ratio))
    return max_bonus * normalized_ratio
