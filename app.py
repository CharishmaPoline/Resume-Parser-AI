import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Resume Parser AI", page_icon="📄", layout="wide")

# ----------------------------
# 🎨 Simple UI theme (Background + cards)
# ----------------------------
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #0b1220 100%);
        color: #e5e7eb;
    }

    /* Titles */
    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: #e5e7eb !important;
    }

    /* Text areas + inputs */
    textarea, input {
        background-color: #0b1220 !important;
        color: #e5e7eb !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #0b1220 !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }

    /* Buttons */
    .stButton > button {
        background: #2563eb !important;
        color: white !important;
        border-radius: 12px !important;
        border: 0px !important;
        padding: 10px 18px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background: #1d4ed8 !important;
    }

    /* Cards-ish look for containers */
    .block-container {
        padding-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Load dataset
# ----------------------------
FILE_NAME = "UpdatedResumeDataSet.csv"
df = pd.read_csv(FILE_NAME)[["Category", "Resume"]].dropna()
df.columns = ["Job Title", "Resume"]

# ----------------------------
# NLTK setup (one time downloads)
# ----------------------------
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemm = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = nltk.word_tokenize(text)
    tokens = [lemm.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# ----------------------------
# ✅ SPEED: cache model training (huge improvement)
# ----------------------------
@st.cache_resource
def train_model(dataframe: pd.DataFrame):
    dataframe = dataframe.copy()
    dataframe["clean_resume"] = dataframe["Resume"].apply(clean_text)
    tfidf_local = TfidfVectorizer(max_features=6000, ngram_range=(1, 2))
    X_local = tfidf_local.fit_transform(dataframe["clean_resume"])
    return dataframe, tfidf_local, X_local

df, tfidf, X = train_model(df)
job_titles = sorted(df["Job Title"].unique().tolist())

# ----------------------------
# Scoring functions
# ----------------------------
def decision(score: float) -> str:
    if score >= 7:
        return "✅ Strong Shortlist"
    elif score >= 5:
        return "⚠️ Maybe"
    else:
        return "❌ Reject"

def score_resume_vs_title(resume_text: str, title: str, top_k: int = 5) -> float:
    v = tfidf.transform([clean_text(resume_text)])

    idx = (df["Job Title"] == title).to_numpy()
    if idx.sum() == 0:
        return 0.0

    job_matrix = X[idx]
    if job_matrix.shape[0] == 0:
        return 0.0

    sims = cosine_similarity(v, job_matrix).ravel()
    if sims.size == 0:
        return 0.0

    top_k = min(top_k, len(sims))
    best = np.sort(sims)[-top_k:]
    return round(float(best.mean() * 10), 2)

def score_resume_vs_jd(resume_text: str, jd_text: str) -> float:
    v_resume = tfidf.transform([clean_text(resume_text)])
    v_jd = tfidf.transform([clean_text(jd_text)])
    sim = cosine_similarity(v_resume, v_jd)[0][0]
    return round(float(sim * 10), 2)

def top_matches_all_titles(resume_text: str, topn: int = 3):
    scores = [(t, score_resume_vs_title(resume_text, t, top_k=5)) for t in job_titles]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topn]

# ----------------------------
# UI
# ----------------------------
st.title("📄 Resume Parser AI (Multi Job Titles + JD Match)")
st.caption("Fast + clean UI (cached model)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Resume Input")
    resume_text = st.text_area("Paste Resume Here", height=360)

with col2:
    st.subheader("Job Titles")

    # ✅ show total job titles
    st.write(f"Total Job Titles: **{len(job_titles)}**")

    show_all = st.checkbox("Show all job titles", value=True)

    search = st.text_input("Search job title (type keyword, ex: data, java, python...)")

    if show_all:
        filtered = job_titles
    else:
        filtered = [t for t in job_titles if search.lower() in t.lower()] if search else job_titles

    # ✅ If search gives 0 results
    if len(filtered) == 0:
        st.warning("No matching job titles found. Try another keyword.")
        filtered = job_titles

    st.write(f"Showing: **{len(filtered)}** job titles")

    selected_title = st.selectbox("Select Job Title", filtered)

st.divider()

st.subheader("Job Description (Optional)")
jd_text = st.text_area("Paste JD Here (optional). If pasted, you'll also get Resume vs JD score.", height=200)

st.divider()

if st.button("Check Match", type="primary"):
    if len(resume_text.strip()) < 50:
        st.error("Please paste a proper resume (min 50 characters).")
    else:
        st.markdown("## ✅ Results")

        title_score = score_resume_vs_title(resume_text, selected_title, top_k=5)
        st.write(f"**Selected Title:** {selected_title}")
        st.write(f"**Resume vs Job Title Score:** {title_score} / 10")
        st.write(f"**Decision:** {decision(title_score)}")

        st.markdown("### 🔍 Top 3 Best Matching Titles (from dataset)")
        for t, sc in top_matches_all_titles(resume_text, topn=3):
            st.write(f"**{t}** → {sc}/10 ({decision(sc)})")

        if len(jd_text.strip()) >= 50:
            jd_score = score_resume_vs_jd(resume_text, jd_text)
            st.markdown("### 🧾 Resume vs Job Description (JD) Match")
            st.write(f"**Resume vs JD Score:** {jd_score} / 10")
            st.write(f"**Decision:** {decision(jd_score)}")
        else:
            st.info("JD paste cheyyakapothe resume vs JD score skip chestadi. (Optional)")
