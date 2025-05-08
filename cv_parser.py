import re
from pathlib import Path
from typing import Set, List, Tuple
import PyPDF2
import pypdf # Ensure pypdf is imported
from smartjobcraft_day1 import SkillNormalizer

# Predefined list of common technical skills 
KNOWN_SKILLS = {
    "python", "sql", "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn",
    "tensorflow", "pytorch", "opencv", "mediapipe", "fastapi", "docker", "aws",
    "spark", "airflow", "kafka", "postgresql", "sqlite", "duckdb", "mongodb",
    "git", "github", "jupyter notebook", "excel", "tableau", "power bi",
    "machine learning", "data analysis", "feature engineering", "model deployment",
    "agile", "scrum", "kanban", "terraform", "c#", "unity", "godot", "blender",
    "autocad", "deep learning", "data science", "big data", "cloud computing",
    "javascript", "r"
}

# Predefined soft skills
SOFT_SKILLS = {
    "analytical mindset", "team player", "communication", "problem-solving",
    "adaptability", "curiosity", "proactivity", "responsibility"
}

# Predefined languages
KNOWN_LANGUAGES = {"english", "spanish", "turkish"}

# Regex for tokenization (words with letters, optionally including numbers or symbols)
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ0-9\+\#\.\-]*")

def extract_text_from_pdf(cv_path: Path) -> str:
    """Extract raw text from a PDF CV."""
    with open(cv_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def extract_skills_and_languages(cv_path: Path, normalizer: SkillNormalizer) -> Tuple[Set[str], Set[str]]:
    """
    Extract technical skills, soft skills, and languages from a CV PDF.
    
    Args:
        cv_path: Path to the CV PDF file
        normalizer: SkillNormalizer instance to standardize skills
    
    Returns:
        Tuple of (technical_skills, languages)
    """
    raw_text = extract_text_from_pdf(cv_path)
    tokens = {t.lower() for t in _TOKEN_RE.findall(raw_text)}
    
    technical_skills = set()
    languages = set()
    
    # Extract technical and soft skills
    for token in tokens:
        if re.fullmatch(r"\d+[\-\d]*", token) or token.startswith("+"):
            continue
        canonical = normalizer.canonical(token)
        if canonical.lower() in KNOWN_SKILLS or token.lower() in KNOWN_SKILLS:
            technical_skills.add(canonical)
        elif canonical.lower() in SOFT_SKILLS or token.lower() in SOFT_SKILLS:
            # Add soft skills to the main skills set for now
            # Later, Candidate dataclass can be updated to have a separate soft_skills field if needed
            technical_skills.add(canonical) 
    
    # Extract languages from the "Languages" section or general text
    # A more robust way would be to look for a specific "Languages" section header
    language_section_match = re.search(r"(Languages|Idiomas|Diller)\s*([\s\S]*?)(?:Technical Skills|Projects|Experience|Education|$)", raw_text, re.IGNORECASE)
    text_to_search_langs = raw_text # Default to searching the whole text
    if language_section_match:
        text_to_search_langs = language_section_match.group(2) # Search within the identified section

    for lang in KNOWN_LANGUAGES:
        # Search for the language name, optionally followed by proficiency levels
        # e.g., "English (C1)", "Spanish: Native", "Turkish - Fluent"
        if re.search(r"\b" + lang + r"\b(?:\s*\(?[BCEAP][12][CEFR]?\)?|\s*:\s*(?:Native|Proficient|Fluent|Advanced|Intermediate|Beginner|Nativo|Fluido|Avanzado|Intermedio|Principiante|Básico))?", text_to_search_langs, re.IGNORECASE):
            languages.add(lang.capitalize())
            
    # If no languages found in a specific section, try a broader search in the whole CV text
    if not languages:
        for lang in KNOWN_LANGUAGES:
            if re.search(r"\b" + lang + r"\b", raw_text, re.IGNORECASE):
                 languages.add(lang.capitalize())

    return technical_skills, languages

def extract_cv_projects(cv_path: Path) -> list[str]:
    """Extracts project sections as text from a CV PDF file."""
    text = ""
    try:
        reader = pypdf.PdfReader(cv_path)
        for page in reader.pages:
            extracted_page_text = page.extract_text()
            if extracted_page_text:
                text += extracted_page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF {cv_path}: {e}")
        return []

    projects_text = []
    current_project_lines = []
    in_section = False
    # Adapt the heading for "Projects", "My Projects", "EXPERIENCES AND PROJECTS", etc., in your CV here.
    # These keywords indicate the start of the project section.
    section_start_keywords = [
        "projects", "my projects", "experiences and projects", "professional experience", 
        "work experience", "experience"
    ] # Lowercase

    # A more general regex to capture project titles or significant project starts.
    # Examples: "Project Name:", "Project Title -", "• Project Name", "Company Name @ Position"
    # This regex targets lines that usually start with a capital letter or have a specific format.
    # You can improve this regex based on the format of project titles in your CV.
    # Example: "E-commerce Platform", "LinguaLink AI", "Big Data Analysis Tool"
    # Or those indicating positions: "Software Engineer at Company"
    main_project_regex = re.compile(
        r"^\s*([A-ZÀ-ÖØ-Þ][\w\sÀ-ÖØ-öø-ÿ\.\(\)\-,]+)(?::|\s*-|\s*•|\s+@\s+|\s+\|)",
        re.IGNORECASE # Case-insensitive
    )
    # Alternatively, you can directly add known project names from your CV:
    # known_project_titles_regex = r"^(E-commerce Platform|LinguaLink AI|Big Data Analysis Tool|SmartJob Craft)"
    # main_project_regex = re.compile(known_project_titles_regex, re.IGNORECASE)


    # Keywords that follow the project section and indicate its end.
    section_end_keywords = [
        "technical skills", "skills", "abilities", "competencies", "education", 
        "certifications", "languages", "courses",
        "publications", "awards", "references",
        "contact", "summary", "profile", "about me", "personal details"
    ] # Lowercase

    lines = text.splitlines()
    project_started_in_section = False

    for line_number, line in enumerate(lines):
        line_strip = line.strip()
        line_lower = line_strip.lower()

        if not in_section:
            if any(keyword in line_lower for keyword in section_start_keywords):
                in_section = True
                # If the section start line is also a project title, capture it
                if main_project_regex.match(line_strip):
                    current_project_lines.append(line_strip)
                    project_started_in_section = True
            continue

        # We are in the project section
        is_potential_end = any(keyword in line_lower for keyword in section_end_keywords)
        
        # If we find a section end keyword AND this line is not a project title
        # AND we are already collecting a project, then end the section.
        if is_potential_end and not main_project_regex.match(line_strip) and current_project_lines:
            # Check the few lines before the end keyword to confirm if it's truly the end of the section.
            # Sometimes these keywords can appear within project descriptions.
            # For example, "Managed skills development for the team."
            # This simple check looks if the keyword is at the beginning of the line.
            if any(line_lower.startswith(keyword) for keyword in section_end_keywords):
                project_text = "\n".join(current_project_lines).strip()
                if project_text:
                    projects_text.append(project_text)
                current_project_lines = []
                in_section = False # Project section ended, can move to other sections.
                project_started_in_section = False
                break # Exit loop as project section has ended

        is_main_project_line = bool(main_project_regex.match(line_strip))

        if is_main_project_line:
            if current_project_lines: # Save the previous project
                project_text = "\n".join(current_project_lines).strip()
                if project_text:
                    projects_text.append(project_text)
            current_project_lines = [line_strip] # Start the new project
            project_started_in_section = True
        elif project_started_in_section and line_strip: # Add meaningful lines after finding a project title
            # We can skip very short lines or lines consisting only of numbers (optional)
            if len(line_strip) > 3 or line_strip.isalpha(): # If not only numbers or long enough
                 current_project_lines.append(line_strip)
    
    # Add the last remaining project after the loop
    if current_project_lines:
        project_text = "\n".join(current_project_lines).strip()
        if project_text:
            projects_text.append(project_text)

    if not projects_text:
        print("DEBUG cv_parser: Could not extract project text blocks from CV. Check regex and keywords.")
    else:
        print(f"DEBUG cv_parser: Extracted {len(projects_text)} project text blocks from CV.")
    return projects_text

if __name__ == "__main__":
    # Quick test
    normalizer = SkillNormalizer("tests/skills_variants.csv")
    cv_path = Path("Kaan_AK_CV.pdf") # Replace with your CV file name
    skills, languages = extract_skills_and_languages(cv_path, normalizer)
    print("Technical Skills Found:", ", ".join(sorted(skills)))
    print("Languages Found:", ", ".join(sorted(languages)))

    projects = extract_cv_projects(cv_path)
    if projects:
        print("\n--- Extracted Projects ---")
        for i, proj in enumerate(projects):
            print(f"Project {i+1}:\n{proj}\n--------------------")
    else:
        print("\nNo projects extracted or found in the CV.")