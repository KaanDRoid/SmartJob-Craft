"""
smartjobcraft_gaprec.py  –  Day 2: GapRecommender
-------------------------------------------------
• summary.csv (Day 1.5) içindeki DÜŞÜK skorlu (<0.60) ilanlar için
  5 maddelik “mikro-öğrenme” planı üretir.
• Çıktılar:  ./learning_plans/<file_id>.md   (markdown)
"""
from __future__ import annotations
import csv, json, os, textwrap
from pathlib import Path
from openai import OpenAI # Ensure OpenAI is imported

THRESHOLD = 0.60
USE_OPENAI = True # Set to False for dummy mode if needed

# LLM back-end
def _dummy_llm(prompt: str) -> str:
    """Her beceri için 1 satırlık blog/YouTube önerisi döndürür (offline test)."""
    missing_text = prompt.split("MISSING:")[-1].split("\n")[0] # Get text after MISSING:
    missing = [s.strip() for s in missing_text.split(",") if s.strip()]
    bullets = [f"- **{skill}**: resmi dokümantasyon + 30 dk YouTube crash-course" for skill in missing[:5]]
    return json.dumps({"plan": "\n".join(bullets)})

if USE_OPENAI:
    # Ensure OPENAI_API_KEY is set in your environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback or error for missing API key
        print("Warning: OPENAI_API_KEY not set. Falling back to dummy LLM.")
        LLM = _dummy_llm
    else:
        client = OpenAI(api_key=api_key)
        def _openai_llm(prompt: str) -> str:
            try:
                rsp = client.chat.completions.create(
                    model="gpt-4o-mini", # Using a cost-effective and capable model
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
                return rsp.choices[0].message.content
            except Exception as e:
                # Log the error or handle it as per application requirements
                print(f"OpenAI API call failed: {e}")
                # Return a JSON-formatted error message that the main function can parse
                return json.dumps({"plan": f"*Error*: Could not generate plan due to API issue: {str(e)}"})
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
            # Strip BOM and whitespace from keys, convert score to float
            row = {k.lstrip('\ufeff').strip(): v for k, v in row.items()}
            try:
                row["score"] = float(row["score"])
            except ValueError:
                print(f"Warning: Could not convert score '{row['score']}' to float for row: {row}")
                row["score"] = 0.0 # Default score if conversion fails
            yield row

def main(summary_csv="summary.csv"):
    plans_dir = Path("learning_plans")
    plans_dir.mkdir(exist_ok=True)

    for row in load_summary(Path(summary_csv)):
        if row["score"] >= THRESHOLD or not row.get("gaps", "").strip(): # Use .get for gaps
            continue

        gaps = [g.strip() for g in row["gaps"].split(",") if g.strip()]
        if not gaps: # Skip if, after stripping, there are no gaps
            continue
            
        prompt_text = PROMPT.format(title=row.get("title", "N/A"), missing=", ".join(gaps))

        try:
            response_str = LLM(prompt_text)
            # Ensure the response is valid JSON before parsing
            if not response_str.strip().startswith("{"):
                 raise json.JSONDecodeError("LLM response is not JSON", response_str, 0)
            
            plan_data = json.loads(response_str)
            md = plan_data.get("plan", "*Error*: Plan not found in LLM response.").strip()
        except json.JSONDecodeError as exc:
            md = f"*Error*: Failed to parse LLM response: {exc}. Response: {response_str[:200]}..."
        except Exception as exc: # Catch any other unexpected errors
            md = f"*Error*: An unexpected error occurred: {exc}"

        out_path = plans_dir / f"{row.get('file_id', 'unknown_file')}.md"
        out_path.write_text(md, encoding="utf-8")
        print(f"✅  {out_path.name}  ({len(gaps)} gaps)")

if __name__ == "__main__":
    # Determine the correct path for summary.csv, assuming it's in the parent directory of the script if run directly
    # Or in the current working directory if the script is run from the project root.
    # For this example, let's assume summary.csv is in the same directory as the script or project root.
    main(summary_csv="summary.csv")
