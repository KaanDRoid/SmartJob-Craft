from typing import Tuple, List
import re
from smartjobcraft_day1 import Job, Candidate
from smartjobcraft_day1 import SkillNormalizer

_RX_TOKEN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\+\#\.\-]{2,}")

def _tokens(text: str) -> List[str]:
    """Basit tokenizasyon → alfa‑num + '.' '+' '#' karakterlerini korur."""
    return _RX_TOKEN.findall(text)

def analyze_projects(projects: list[str], normalizer: SkillNormalizer):
    if not projects:
        print("Uyarı: CV'de hiç proje bulunamadı. Lütfen projelerinizi ekleyin.")
        return

    for proj in projects:
        tokens = [t.lower() for t in _tokens(proj)]
        proj_skills = {normalizer.canonical(t).lower() for t in tokens}
        if not proj_skills:
            print(f"Uyarı: '{proj}' projesinde beceri tespit edilmedi.")
            print("Öneri: Bu projede kullanılan becerileri ekleyin (örneğin, Python, SQL).")

def rank_projects(
    job: Job,
    candidate: Candidate,
    normalizer: SkillNormalizer
) -> Tuple[str, float]:
    """
    Rank the candidate's projects against the job's must-have and nice-to-have skills.

    Returns:
      top_project: the project string with highest coverage
      coverage: fraction of skills covered (0–1), weighted as:
                (|must_match| + 0.5·|nice_match|) / (|must_have| + 0.5·|nice_to_have|)
    """
    must = {normalizer.canonical(s).lower() for s in job.must_have or []}
    nice = {normalizer.canonical(s).lower() for s in job.nice_to_have or []}
    
    if not must and not nice:
        return "", 0.0

    analyze_projects(candidate.projects, normalizer)

    top_proj = ""
    best_cov = 0.0
    stop_words = {'ile', 've', 'yaptım.', 'analizi', 'bir', 'için'}
    for proj in candidate.projects:
        tokens = [t.lower() for t in _tokens(proj) if t.lower() not in stop_words]
        proj_skills = {normalizer.canonical(t).lower() for t in tokens}
        matched_must = must & proj_skills
        matched_nice = nice & proj_skills
        cov = (
            len(matched_must) + 0.5 * len(matched_nice)
        ) / (len(must) + 0.5 * len(nice) or 1)
        print(f"Proje: {proj}")
        print(f"Çıkarılan Beceriler: {proj_skills}")
        print(f"İş Becerileri (Must): {must}")
        print(f"İş Becerileri (Nice): {nice}")
        print(f"Kesişim (Must): {matched_must}")
        print(f"Kesişim (Nice): {matched_nice}")
        print(f"Kapsama Skoru: {cov}")
        if cov > best_cov:
            best_cov = cov
            top_proj = proj
    return top_proj, best_cov

def compute_project_bonus(coverage: float) -> float:
    """
    Compute a project relevance bonus based on coverage.

    Returns 0.05 if any must-have skill is covered, else 0.
    """
    return 0.05 if coverage > 0 else 0.0

# CLI demonstration
if __name__ == "__main__":
    # Minimal demonstration with dummy data
    from smartjobcraft_day1 import _dummy_llm, SkillNormalizer, JobParser
    import glob
    import PyPDF2

    def extract_projects(cv_path: str) -> list[str]:
        # Read PDF text
        with open(cv_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted: # Check if text extraction was successful
                    text += extracted + "\n"

        print("--- PDF'den Çıkan Metin ---")
        # print(text) # Keep this commented unless needed for deep debugging

        projects_text = []
        current_project_lines = []
        in_section = False
        found_first_project = False
        lines = text.splitlines()

        # Regex to identify the start of the Projects section
        section_start_regex = re.compile(r"^\s*Projects\s*$", re.IGNORECASE)
        
        # Improved regex to identify ONLY main project titles, not sub-headers
        main_project_regex = re.compile(
            r"^(E-commerce|LinguaLink|Big Data).*?:",
            re.IGNORECASE
        )
        
        # Regex to identify potential section end markers
        section_end_regex = re.compile(r"^\s*Technical Skills\s*$", re.IGNORECASE)

        for i, line in enumerate(lines):
            line_strip = line.strip()
            
            # 1. Find the Projects section
            if not in_section:
                if "Projects" in line_strip:
                    in_section = True
                continue  # Skip until we find the Projects section

            # 2. We're in the Projects section
            # Check if this line starts a new main project
            is_main_project = bool(main_project_regex.match(line))
            is_section_end = bool(section_end_regex.match(line_strip))
            
            if is_main_project or is_section_end:
                # If we already found a project, save the current one before starting the next
                if current_project_lines:
                    project_text = "\n".join(current_project_lines).strip()
                    if project_text:  # Only add non-empty projects
                        projects_text.append(project_text)
                    # Reset for the next project
                    current_project_lines = []
                
                if is_section_end:
                    break  # End of the Projects section
                
                # Start collecting the new project
                if is_main_project:
                    current_project_lines.append(line)
                    found_first_project = True
            
            # Add lines to the current project if we're already tracking one
            elif found_first_project:
                current_project_lines.append(line)

        # Don't forget to add the last project if there is one
        if current_project_lines:
            project_text = "\n".join(current_project_lines).strip()
            if project_text:
                projects_text.append(project_text)

        if not projects_text:
            print("DEBUG: No project text blocks extracted.")
        else:
            print(f"DEBUG: Extracted {len(projects_text)} project text blocks.")

        return projects_text

    parser = JobParser(_dummy_llm) # Use dummy for CLI demo speed
    normalizer = SkillNormalizer("tests/skills_variants.csv")
    # Load first ad
    path = glob.glob("tests/spanish_ad_*.txt")[0]
    raw = open(path, encoding="utf-8").read()
    job = parser.parse(raw)
    # Sample candidate with real CV
    cv_path = "Kaan_AK_CV.pdf"  # CV dosya yolunu güncelle
    projects = extract_projects(cv_path)
    candidate = Candidate(
        name="Sample",
        headline="",
        skills=set(),
        projects=projects,
        education=[],
    )
    top, cov = rank_projects(job, candidate, normalizer)
    bonus = compute_project_bonus(cov)
    print(f"Top project: {top}\nCoverage: {cov:.2f}\nBonus: {bonus:.2f}")