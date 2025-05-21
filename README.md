# SmartJob-Craft

SmartJob-Craft is an AI-powered platform that helps job seekers optimize their applications by analyzing CVs and job ads, identifying skills gaps, ranking relevant projects, and generating personalized cover letters and learning plans. The goal is to make job applications smarter, more relevant, and more successful—while ensuring privacy and data protection.



---

## Features

- **AI-Powered Skill Matching:**  
  Parses job ads and CVs, matches candidate skills to job requirements, and computes a relevance score using LLMs (OpenAI GPT-4o or Mistral-7B-Instruct).
- **Project Ranking:**  
  Automatically extracts and ranks your CV projects in relation to each job posting.
- **Skill Gap Analysis & Learning Plans:**  
  Identifies missing skills and generates actionable micro-learning plans for gap closure.
- **Personalized Cover Letters:**  
  Generates tailored cover letters in Markdown format, ready to use for your applications.
- **Portfolio PDF Generator:**  
  Renders a one-page, visually appealing portfolio (with QR code to your GitHub) as a PDF.
- **Streamlit UI:**  
  User-friendly, wizard-style web interface for uploading CVs, reviewing matches, and downloading results.
- **Privacy First:**  
  All processing is in-memory; data is purged after your session. Anonymized skill logs are auto-deleted after 14 days.

---

## Tech Stack

- **Python** (core logic & backend)
- **Streamlit** (frontend UI)
- **OpenAI GPT-4o / Mistral-7B-Instruct** (LLM job ad/CV analysis)
- **LangChain** (LLM orchestration)
- **PyPDF2, python-reportlab** (PDF parsing & rendering)
- **Pandas** (scoring summary & CSV export)

---

## Getting Started

1. **Clone the Repository**
    ```bash
    git clone https://github.com/KaanDRoid/SmartJob-Craft.git
    cd SmartJob-Craft
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit App**
    ```bash
    streamlit run smartjobcraft_ui.py
    ```

4. **Follow the 4-Step Wizard:**
    1. Upload your CV (PDF)
    2. Paste a job advertisement
    3. Review your skill match, project ranking, and learning recommendations
    4. Download your personalized portfolio PDF and cover letter

---

## Privacy & Data Usage

- All processing happens in-memory—your CV and job ads are never stored permanently.

---

## Roadmap

- [x] Skill extraction and normalization (multilingual)
- [x] Job ad parsing (visa, remote, contract parsing)
- [x] Project ranking and bonus scoring
- [x] Learning plan generation for skill gaps
- [x] Cover letter and PDF portfolio generator
- [ ] Improved mobile/responsive UI (i mean i hope)
- [ ] More advanced analytics and recruiter view

---

## Author

This project was developed by me [KaanDRoid](https://github.com/KaanDRoid).  
Feel free to open issues or suggestions!

---

## License

This project is licensed under the Apache 2.0 License.

---
