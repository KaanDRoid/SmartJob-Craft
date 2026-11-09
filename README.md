# SmartJob-Craft

SmartJob-Craft compares a CV (PDF) with a job ad. It extracts skills and languages from the CV, uses the OpenAI API to turn the job ad into structured JSON (title, company, must-have and nice-to-have skills), scores how well the two match, ranks the CV projects against the job, suggests a short learning plan for missing skills and drafts a cover letter. There is a Streamlit UI for a single CV/job pair and a few batch scripts for scoring many ads at once.

This is an early MVP built as a personal project.

## How it works

1. **CV parsing** (`cv_parser.py`): text is read with PyPDF2, tokenized and matched against a fixed list of technical skills, soft skills and languages (English, Spanish, Turkish).
2. **Skill normalization** (`SkillNormalizer` in `smartjobcraft_day1.py`): skill variants are mapped to one canonical name using `tests/skills_variants.csv` (e.g. "ML" or "makine öğrenimi" -> Machine Learning), with a rapidfuzz fallback (WRatio >= 90).
3. **Job parsing** (`JobParser`): a few-shot prompt sent to `gpt-4o-mini` returns the job as JSON.
4. **Scoring** (`score_candidate`):
   ```
   base  = 0.7 * (matched must-have / all job skills) + 0.3 * (matched nice-to-have / nice-to-have)
   score = base + exp_bonus + project_bonus
   ```
   The batch scripts add an experience bonus (0.02 per year above the requirement, max 0.1) and an education bonus (0.03). The project bonus is 0.05: in the UI when the best project covers more than 10% of the job skills, in `smartjobcraft_day2.py` for any coverage above zero.
5. **Project ranking** (`smartjobcraft_projectranker.py`): each project is scored by weighted coverage of must-have (weight 1) and nice-to-have (weight 0.5) skills.
6. **Gap plan** (`smartjobcraft_gaprec.py`): for scores below 0.60, the LLM writes a 5-bullet micro-learning plan for the missing skills.
7. **Cover letter** (`smartjobcraft_coverletter.py`): a short letter (under 200 words) that mentions the top project and the must-have skills. The UI can export it as PDF with fpdf2.

The UI also has a small "career coach" chat in the sidebar.

## Setup

Requires Python 3.10+ and an OpenAI API key.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...        # PowerShell: $env:OPENAI_API_KEY="..."
streamlit run smartjobcraft_ui.py
```

In the UI: upload a CV PDF, paste a job ad, then press "Match and Score".

The uploaded CV is written to `tmp_cv.pdf` in the working directory, API calls are logged to `logs/api_calls.log`, and the cover letter PDF is saved as `coverletter.pdf`. All of these are git-ignored.

## Batch scripts

The batch scripts expect a CV at `Kaan_AK_CV.pdf` and job ads as text files in `tests/` (`spanish_ad_*.txt`, `turkish_ad_*.txt`, `english_ad_*.txt`; several ads in one file can be separated with `# ilan` headings). The ad files and the CV are not included in the repo.

| Script | What it does | Output |
| --- | --- | --- |
| `smartjobcraft_analysis.py` | Scores every ad against a hardcoded skill set | `job_json/`, `summary.csv` |
| `smartjobcraft_day1_5.py` | Same, but takes skills from the CV and adds experience/education bonuses | `summary.csv` |
| `smartjobcraft_gaprec.py` | Learning plans for low-scoring rows of `summary.csv` | `learning_plans/` |
| `smartjobcraft_day2.py` | Full pipeline: scoring, project ranking, learning plans | `job_json2/`, `learning_plans/`, `summary2.csv` |
| `smartjobcraft_coverletter.py` | Cover letters for rows of `summary2.csv` with score < 0.6 and a top project | `cover_letters/` |

`smartjobcraft_day1.py` holds the shared classes; running it directly parses the test ads as a quick check. Set `USE_DUMMY=1` when running `smartjobcraft_day2.py` to skip the API for job parsing (learning plans still use it).

## Tests

```bash
pytest tests
```

`tests/test_jobparser.py` checks parsing and scoring with a dummy LLM, so it needs no API key.

## Known limitations

- Project extraction in the UI (`smartjobcraft_projectranker.extract_projects`) matches project titles from my own CV, so other CVs will usually show "no projects found". `cv_parser.extract_cv_projects` is a more general but looser heuristic.
- Skill extraction only finds skills from the fixed lists in `cv_parser.py`.
- Candidate experience is hardcoded to 3 years in the batch scripts.

## Repo layout

```
smartjobcraft_ui.py             Streamlit app
smartjobcraft_day1.py           Job/Candidate models, JobParser, SkillNormalizer, scoring
cv_parser.py                    CV text, skill, language and project extraction
smartjobcraft_projectranker.py  Project extraction and ranking used by the UI
smartjobcraft_gaprec.py         Learning plan prompt and batch script
smartjobcraft_coverletter.py    Cover letter prompt and batch script
smartjobcraft_analysis.py       Batch scoring (fixed skill set)
smartjobcraft_day1_5.py         Batch scoring (skills from CV)
smartjobcraft_day2.py           Batch pipeline with projects and learning plans
tests/skills_variants.csv       Skill synonyms (EN/TR/ES)
tests/test_jobparser.py         Unit test
requirements.txt
```

## License

MIT, see [LICENSE](LICENSE).
