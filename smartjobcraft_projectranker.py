from typing import Tuple, List, Set # Set eklendi
import re
from smartjobcraft_day1 import Job, Candidate, SkillNormalizer # SkillNormalizer import edildi
import PyPDF2 # PyPDF2 import edildi

_RX_TOKEN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\+\#\.\-]{2,}")

def _tokens(text: str) -> List[str]:
    """Basit tokenizasyon → alfa‑num + '.' '+' '#' karakterlerini korur."""
    return _RX_TOKEN.findall(text)

def extract_projects(cv_path: str) -> list[str]:
    """Extract project descriptions from the CV's Projects section."""
    with open(cv_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

    projects_text = []
    current_project_lines = []
    in_section = False
    lines = text.splitlines()

    # Regex to identify the start of the Projects section
    section_start_regex = re.compile(r"^\s*Projects\s*$", re.IGNORECASE)
    
    # Regex to identify project titles (specific to Kaan_AK_CV.pdf structure)
    project_title_regex = re.compile(
        r"^(E-commerce Customer Spending Analysis|LinguaLink: Real Time ASL|Big Data Infrastructure Exercise).*?",
        re.IGNORECASE
    )
    
    # Regex to identify potential section end markers
    section_end_regex = re.compile(r"^\s*(Technical Skills|Education|Experience|Languages|Certifications)\s*$", re.IGNORECASE)

    for line in lines:
        line_strip = line.strip()
        
        if not in_section:
            if section_start_regex.match(line_strip):
                in_section = True
            continue # Continue until "Projects" section is found

        # If a section end marker is found, process the last project and stop.
        if section_end_regex.match(line_strip):
            if current_project_lines:
                project_text = "\n".join(current_project_lines).strip()
                if project_text: # Ensure not adding empty strings
                    projects_text.append(project_text)
            current_project_lines = [] # Reset for safety, though we break
            break # Exit project extraction once a new section starts

        # If a new project title is found
        if project_title_regex.match(line_strip):
            if current_project_lines: # Save the previous project
                project_text = "\n".join(current_project_lines).strip()
                if project_text:
                    projects_text.append(project_text)
            current_project_lines = [line_strip] # Start new project with its title line
        elif in_section and current_project_lines: # If in section and a project has started, append current line
            current_project_lines.append(line_strip)

    # Add the last collected project after the loop if any lines were collected
    if current_project_lines:
        project_text = "\n".join(current_project_lines).strip()
        if project_text:
            projects_text.append(project_text)
    
    # Fallback if no projects are extracted by the specific regexes
    return projects_text if projects_text else ["No specific projects extracted, check regex or CV structure."]

def rank_projects(
    job: Job,
    candidate: Candidate,
    normalizer: SkillNormalizer
) -> Tuple[str, float]:
    """
    Rank the candidate's projects against the job's must-have and nice-to-have skills.
    """
    must = {normalizer.canonical(s).lower() for s in job.must_have or []}
    nice = {normalizer.canonical(s).lower() for s in job.nice_to_have or []}
    
    if not must and not nice:
        return "", 0.0

    top_proj = ""
    best_cov = 0.0
    # More comprehensive stop words might be beneficial depending on project description style
    stop_words = {'a', 'an', 'the', 'is', 'was', 'were', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'and', 'or', 'but', 'if', 'as', 'my', 'this', 'that', 'project', 'developed', 'using', 'ile', 've', 'yaptım.', 'analizi', 'bir', 'için'}
    
    for proj_text in candidate.projects:
        # Tokenize and normalize skills from the project description
        tokens = [t.lower() for t in _tokens(proj_text) if t.lower() not in stop_words and len(t) > 1]
        proj_skills = {normalizer.canonical(t).lower() for t in tokens}
        
        matched_must = must & proj_skills
        matched_nice = nice & proj_skills
        
        # Weighted coverage calculation
        denominator = (len(must) + 0.5 * len(nice))
        if denominator == 0: # Avoid division by zero if no skills in job ad
            cov = 0.0
        else:
            cov = (len(matched_must) + 0.5 * len(matched_nice)) / denominator
            
        if cov > best_cov:
            best_cov = cov
            top_proj = proj_text # Store the full project text
    
    return top_proj, best_cov

def compute_project_bonus(coverage: float) -> float:
    """Compute a project relevance bonus based on coverage."""
    # Bonus can be scaled based on coverage quality if desired
    return 0.05 if coverage > 0.1 else 0.0 # Give bonus if at least 10% coverage

# CLI demonstration
if __name__ == "__main__":
    # Dummy JobParser and SkillNormalizer for testing
    class DummyJobParser:
        def parse(self, ad_text):
            # Simulate parsing a job ad
            return Job(title="Test Job", company="TestCo", location="TestLocation", contract=None,
                       must_have=["Python", "SQL", "AWS"], nice_to_have=["Spark", "Docker"],
                       language_requirement=["English"])

    parser = DummyJobParser() # Use the dummy parser
    normalizer = SkillNormalizer("tests/skills_variants.csv") # Ensure this path is correct
    
    # Example: Load a job ad (or use a dummy one)
    # For a real test, you might load an ad from a file as in smartjobcraft_day1.py
    # For this example, we use the dummy job from DummyJobParser
    job = parser.parse("dummy ad text") 

    cv_path = "Kaan_AK_CV.pdf" # Ensure Kaan_AK_CV.pdf is in the script's directory or provide full path
    if not Path(cv_path).exists():
        print(f"CV file not found: {cv_path}")
    else:
        projects = extract_projects(cv_path)
        if not projects or projects == ["No specific projects extracted, check regex or CV structure."]:
            print(f"No projects extracted from {cv_path}. Check CV content and extract_projects regex.")
            # Assign some dummy projects if none are found, for testing rank_projects
            projects = [
                "E-commerce Customer Spending Analysis: Analyzed clothing e-commerce and in store session data using Python (Pandas, Matplotlib, Seaborn), SQL, and Tableau to identify spending patterns and provide actionable insights for marketing strategies.",
                "LinguaLink: Real Time ASL ↔ Voice Translator: Developing an AI driven addon that translates American Sign Language to voice and vice versa in real time using Python (TensorFlow, OpenCV, MediaPipe).",
                "Big Data Infrastructure Exercise: Completed a series of hands-on exercises on big data storage, processing, and analysis using tools like Hadoop, Spark, Kafka, and Hive. Focused on building scalable data pipelines and performing distributed computations."
            ]

        # Create a dummy Candidate object
        candidate = Candidate(
            name="Test Candidate",
            headline="Data Enthusiast",
            skills={"Python", "SQL", "AWS", "Tableau", "TensorFlow", "OpenCV"}, # Example skills
            projects=projects,
            education=["M.Sc. Big Data & AI"],
        )

        top_project, coverage = rank_projects(job, candidate, normalizer)
        bonus = compute_project_bonus(coverage)

        print(f"Top Project: {top_project}")
        print(f"Coverage: {coverage:.2f}")
        print(f"Bonus: {bonus:.2f}")