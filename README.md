# Resume Parser AI (Multi Job Titles + JD Match)

## 📌 Project Overview
Recruiters receive hundreds of resumes for a single job role. This project helps automate resume screening using NLP by matching resumes with job titles and job descriptions (JD), and generating a match score (0–10).

## ✅ Features
- Multi-job title resume scoring (25 job categories)
- Searchable job title selection
- Resume vs Job Title match score (0–10)
- Resume vs Job Description (JD) match score
- Top 3 best matching job roles suggestion
- Streamlit web app interface

## 🛠 Tech Stack
- Python
- NLTK
- Scikit-learn (TF-IDF, Cosine Similarity)
- Streamlit

## 📂 Dataset
Kaggle Resume Dataset  
(UpdatedResumeDataSet.csv)

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
