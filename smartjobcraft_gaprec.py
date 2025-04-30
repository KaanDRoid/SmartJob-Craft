from __future__ import annotations
import csv, json, os, textwrap
from pathlib import Path

THRESHOLD = 0.60          # skor eşiği
USE_OPENAI = True         # dummy mod için False yap

# LLM back-end
def _dummy_llm(prompt: str) -> str:
    """Her beceri için 1 satırlık blog/YouTube önerisi döndürür (offline test)."""
    missing = [s.strip() for s in prompt.split("MISSING:")[1].split(",") if s.strip()]
    bullets = [f"- **{skill}**: resmi dokümantasyon + 30 dk YouTube crash-course" for skill in missing[:5]]
    return json.dumps({"plan": "\n".join(bullets)})

if USE_OPENAI:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _openai_llm(prompt: str) -> str:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return rsp.choices[0].message.content

    LLM = _openai_llm
else:
    LLM = _dummy_llm


PROMPT = textwrap.dedent(
"""
You are a senior tech mentor.
The candidate is applying for **{title}** but lacks:

MISSING: {missing}

Create a SHORT micro-learning plan in **exactly 5 bullet points**.
Each bullet = resource  ➜ effort (h).  
Return ONLY JSON: {{ "plan": "<markdown-bullets>" }}
""").strip()

def load_summary(csv_path: Path):
    with csv_path.open(encoding="utf-8-sig") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            row = {k.lstrip('\ufeff').strip(): v for k, v in row.items()}
            row["score"] = float(row["score"])
            yield row

def main(summary_csv="summary.csv"):
    plans_dir = Path("learning_plans");  plans_dir.mkdir(exist_ok=True)

    for row in load_summary(Path(summary_csv)):
        if row["score"] >= THRESHOLD or not row["gaps"].strip():
            continue

        gaps = [g.strip() for g in row["gaps"].split(",") if g.strip()]
        prompt = PROMPT.format(title=row["title"], missing=", ".join(gaps))

        try:
            md = json.loads(LLM(prompt))["plan"].strip()
        except Exception as exc:
            md = f"*Error*: {exc}"

        out = plans_dir / f"{row['file_id']}.md"
        out.write_text(md, encoding="utf-8")
        print(f"✅  {out.name}  ({len(gaps)} gaps)")

if __name__ == "__main__":
    main()
