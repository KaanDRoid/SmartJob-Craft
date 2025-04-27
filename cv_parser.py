from pathlib import Path
import re
from smartjobcraft_day1 import SkillNormalizer

_RX_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\+\#\.\-]{2,}")

def extract_skills(cv_path: Path, normalizer: SkillNormalizer) -> set[str]:

    if cv_path.suffix.lower() == ".pdf":
        import pypdf
        text = "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(cv_path).pages)
    else:
        text = cv_path.read_text(encoding="utf-8", errors="ignore")

    skills = set()
    # Languages header or similar (capture every language, every file)
    for match in re.finditer(r"^\s*(languages?|diller|idiomas)[^:\n]*:\s*(.+)$", text, flags=re.I|re.M):
        for tok in re.split(r"[,/;]", match.group(2)):
            canon = normalizer.canonical(tok.strip())
            if canon:
                skills.add(canon)

    # N-gram extraction (for each file)
    words = _RX_WORD.findall(text.lower())
    unigrams = words
    bigrams  = [" ".join(t) for t in zip(words, words[1:])]
    trigrams = [" ".join(t) for t in zip(words, words[1:], words[2:])]
    phrases  = unigrams + bigrams + trigrams
    skills.update(normalizer.canonical_set(phrases))
    return skills
