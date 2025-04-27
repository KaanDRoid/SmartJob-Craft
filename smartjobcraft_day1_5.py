
from pathlib import Path
import re, os, json, pandas as pd
from smartjobcraft_day1 import JobParser, SkillNormalizer, score_candidate
from cv_parser import extract_skills

# parameters
CV_FILE         = Path("Kaan_AK_CV.pdf")     
ADS_PATTERNS    = ["tests/spanish_ad_*.txt", "tests/turkish_ad_*.txt", "tests/english_ad_*.txt"]
USE_OPENAI      = True
CAND_YEARS_EXP  = 3                     
EDU_KEYWORDS    = {"engineering", "mathematics", "statistics"}

# llms
from smartjobcraft_day1 import _dummy_llm   
if USE_OPENAI:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    def _llm(prompt: str) -> str:
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0,
            response_format={"type":"json_object"},
        ).choices[0].message.content
else:
    _llm = _dummy_llm

# utils
_SPLIT = re.compile(r"^\s*#+\s*i?lan", re.I)
def split_ads(txt:str):
    chunk, out = [], []
    for ln in txt.splitlines():
        if _SPLIT.match(ln):
            if chunk: out.append("\n".join(chunk)); chunk=[]
            continue
        chunk.append(ln)
    if chunk: out.append("\n".join(chunk))
    return [c.strip() for c in out if c.strip()]

_RX_EXP = re.compile(r"(\d+)\s*\+?\s*(?:años|yrs?|years)", re.I)

def min_exp(posting:str)->int:
    m = _RX_EXP.search(posting)
    return int(m.group(1)) if m else 0

def education_bonus(cv_tokens:set[str])->float:
    return 0.03 if EDU_KEYWORDS & {t.lower() for t in cv_tokens} else 0.0

# main
def main():
    norm  = SkillNormalizer("tests/skills_variants.csv")
    cand_skills = extract_skills(CV_FILE, norm)
    parser = JobParser(_llm)

    rows=[]
    for pat in ADS_PATTERNS:
        for path in Path().glob(pat):
            raw = path.read_text(encoding="utf-8")
            for i,ad in enumerate(split_ads(raw),1):
                job = parser.parse(ad)
                exp_bonus = 0.0
                needed = min_exp(ad)
                if needed:
                    diff = max(0, CAND_YEARS_EXP - needed)
                    exp_bonus = min(0.1, diff*0.02)
                edu_bonus = education_bonus(cand_skills)

                score = score_candidate(
                    cand_skills, job,
                    exp_bonus=exp_bonus + edu_bonus
                )
                must = job.must_have or []
                nice = job.nice_to_have or []
                gaps  = set(norm.canonical_set(must + nice)) - cand_skills
                rows.append({
                    "file_id":f"{path.stem}__{i}",
                    "title":job.title,
                    "score":round(score,3),
                    "gaps":", ".join(sorted(gaps)[:7])
                })

    df=pd.DataFrame(sorted(rows,key=lambda r:r["score"],reverse=True))
    print(df.to_string(index=False))
    df.to_csv("summary.csv",index=False,encoding="utf-8-sig")

if __name__=="__main__":
    main()
