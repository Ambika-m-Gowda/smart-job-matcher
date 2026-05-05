import streamlit as st
import pandas as pd
import requests
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔑 API KEY
API_KEY = "3241afbb89mshf592a1e7650c0a1p1d62f6jsne5ae0856152b"

# ------------------ DATA ------------------
JOB_ROLES = [
    "Data Analyst", "Data Scientist", "Data Engineer",
    "Business Analyst", "Machine Learning Engineer",
    "AI Engineer", "BI Analyst", "Software Engineer",
    "Backend Developer", "Frontend Developer",
    "Full Stack Developer", "DevOps Engineer",
    "Cloud Engineer", "SQL Developer", "ETL Developer"
]

CITIES = [
    "Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad",
    "Pune", "Kolkata", "Ahmedabad",
    "New York", "San Francisco", "Chicago",
    "London", "Berlin", "Paris",
    "Toronto", "Sydney", "Singapore", "Dubai"
]

SKILLS = [
    "python", "sql", "excel", "power bi",
    "tableau", "machine learning", "statistics"
]

# ------------------ AUTOCOMPLETE ------------------
def autocomplete(label, options):
    user_input = st.text_input(label)

    if user_input:
        filtered = [o for o in options if user_input.lower() in o.lower()]
        if filtered:
            return st.selectbox("Suggestions", filtered)
        else:
            return user_input
    return ""

# ------------------ FETCH JOBS ------------------
def fetch_jobs(role, location):
    url = "https://jsearch.p.rapidapi.com/search"

    headers = {
        "X-RapidAPI-Key": API_KEY,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    params = {
        "query": f"{role} jobs in {location}",
        "page": "1",
        "num_pages": "1"
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        st.error("API Error")
        return pd.DataFrame()

    data = response.json()

    jobs = []
    for job in data.get("data", []):
        jobs.append({
            "title": job.get("job_title", ""),
            "company": job.get("employer_name", ""),
            "location": job.get("job_city", "") or job.get("job_country", ""),
            "description": job.get("job_description", ""),
            "apply_link": job.get("job_apply_link", "")
        })

    return pd.DataFrame(jobs)

# ------------------ RESUME ------------------
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.lower()

def extract_skills(text):
    return [skill for skill in SKILLS if skill in text]

# ------------------ MATCHING ------------------
def advanced_match(resume_text, resume_skills, job_desc):
    job_desc = job_desc.lower()
    job_skills = [s for s in SKILLS if s in job_desc]

    if not job_skills:
        return 0, []

    skill_score = (len(set(resume_skills).intersection(set(job_skills))) / len(job_skills)) * 50

    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_desc])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    similarity_score = similarity * 30

    keyword_score = (len(resume_skills) / len(SKILLS)) * 20

    total = skill_score + similarity_score + keyword_score
    missing = list(set(job_skills) - set(resume_skills))

    return round(total, 2), missing

# ------------------ UI ------------------
st.title("🚀 Smart Job Matcher")

role = autocomplete("Search Job Role", JOB_ROLES)
location = autocomplete("Search Location", CITIES)

experience = st.slider("Experience (Years)", 0, 10, 0)

if st.button("Fetch Jobs"):
    if not role or not location:
        st.warning("Enter role and location")
    elif "PASTE" in API_KEY:
        st.error("Add API key")
    else:
        jobs_df = fetch_jobs(role, location)

        if jobs_df.empty:
            st.error("No jobs found")
        else:
            st.session_state["jobs"] = jobs_df
            st.success(f"{len(jobs_df)} jobs found")

if "jobs" in st.session_state:
    st.subheader("Job Listings")

    for _, row in st.session_state["jobs"].iterrows():
        st.write(f"### {row['title']}")
        st.write(f"{row['company']} | {row['location']}")

        if str(experience) in row["description"]:
            st.success(f"Matches {experience} years")

        st.markdown(f"[Apply Here]({row['apply_link']})")
        st.write("---")

uploaded_file = st.file_uploader("Upload Resume (Optional)", type=["pdf"])

if uploaded_file and "jobs" in st.session_state:
    resume_text = extract_text(uploaded_file)
    resume_skills = extract_skills(resume_text)

    st.write("Your Skills:", resume_skills)

    for _, row in st.session_state["jobs"].iterrows():
        score, missing = advanced_match(resume_text, resume_skills, row["description"])

        st.write(f"### {row['title']}")
        st.write(f"Match: {score}%")

        if missing:
            st.warning(f"Missing: {', '.join(missing)}")

        st.markdown(f"[Apply Here]({row['apply_link']})")
        st.write("---")