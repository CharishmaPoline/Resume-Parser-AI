import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# ✅ 1) Load dataset
# ------------------------------------------------------------
FILE_NAME = "UpdatedResumeDataSet.csv"  # ✅ your file name
df = pd.read_csv(FILE_NAME)

print("✅ Dataset loaded successfully!")
print("✅ Total rows:", len(df))
print("✅ Column names:", df.columns.tolist())
print("\n✅ Sample data:\n", df.head(3))

# ------------------------------------------------------------
# ✅ 2) Use correct columns (your dataset has: Category, Resume)
# ------------------------------------------------------------
df = df[["Category", "Resume"]].dropna()
df.columns = ["Job Title", "Resume"]

# ------------------------------------------------------------
# ✅ 3) Download NLTK resources (one time)
# ------------------------------------------------------------
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
# optional safe
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemm = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # remove urls
    text = re.sub(r"[^a-z\s]", " ", text)         # keep only letters
    text = re.sub(r"\s+", " ", text).strip()

    tokens = nltk.word_tokenize(text)
    tokens = [lemm.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

print("\n✅ Cleaning resumes...")
df["clean_resume"] = df["Resume"].apply(clean_text)

# ------------------------------------------------------------
# ✅ 4) TF-IDF Vectorization
# ------------------------------------------------------------
print("✅ Creating TF-IDF vectors...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(df["clean_resume"])

# ------------------------------------------------------------
# ✅ 5) Build Job Title profiles automatically (multi job titles)
# ------------------------------------------------------------
print("✅ Building job profiles (auto skills per job)...")
terms = np.array(tfidf.get_feature_names_out())
job_titles = df["Job Title"].unique()

job_vectors = {}   # job title -> average TFIDF vector
job_profiles = {}  # job title -> top terms list

for title in job_titles:
    idx = (df["Job Title"] == title).to_numpy()   # ✅ FIX HERE
    job_vec = X[idx].mean(axis=0)                 # average vector
    job_vectors[title] = job_vec

    mean_tfidf = np.asarray(job_vec).ravel()
    top_ids = mean_tfidf.argsort()[::-1][:30]
    job_profiles[title] = terms[top_ids].tolist()

print("✅ Total job titles found:", len(job_titles))

# ------------------------------------------------------------
# ✅ 6) Score any resume for any job title (0–10)
# ------------------------------------------------------------
def score_resume_for_title(resume_text: str, title: str, top_k: int = 5) -> float:
    clean = clean_text(resume_text)
    v = tfidf.transform([clean])

    idx = (df["Job Title"] == title).to_numpy()
    job_matrix = X[idx]  # all resumes vectors for that title

    sims = cosine_similarity(v, job_matrix).ravel()   # similarity with each resume
    if len(sims) == 0:
        return 0.0

    top_k = min(top_k, len(sims))
    best = np.sort(sims)[-top_k:]          # top-k similarities
    score = best.mean() * 10               # 0..10
    return round(float(score), 2)


def decision(score: float) -> str:
    if score >= 7:
        return "✅ Strong Shortlist"
    elif score >= 5:
        return "⚠️ Maybe"
    else:
        return "❌ Reject"

# ------------------------------------------------------------
# ✅ 7) Demo
# ------------------------------------------------------------
sample_resume = df.iloc[0]["Resume"]
sample_title = df.iloc[0]["Job Title"]

print("\n==============================")
print("✅ DEMO OUTPUT")
print("==============================")
print("Job Title:", sample_title)
demo_score = score_resume_for_title(sample_resume, sample_title)
print("Score:", demo_score, "/ 10")
print("Decision:", decision(demo_score))

print("\nTop skills for this Job Title (auto generated):")
print(job_profiles[sample_title][:15])

# ------------------------------------------------------------
# ✅ 8) Score all resumes + save CSV
# ------------------------------------------------------------
print("\n✅ Scoring all resumes for their own job title...")
df["score"] = df.apply(lambda r: score_resume_for_title(r["Resume"], r["Job Title"]), axis=1)
df["decision"] = df["score"].apply(decision)

out_file = "final_resume_scores.csv"
df.to_csv(out_file, index=False)
print("✅ Saved:", out_file)
