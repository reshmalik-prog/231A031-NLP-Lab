"""model.py - NLP logic: cleaning, TF-IDF, Logistic Regression, urgency rules."""
import re                                                    # regex for cleaning text
import pandas as pd                                          # load CSV dataset
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression

URGENT_WORDS = {"urgent", "urgently", "asap", "emergency", "immediately", "sick", "fire", "unsafe", "danger", "accident",
                "tomorrow", "today", "leakage", "broke", "blocked", "insect", "rash"}

def clean_text(text):
    text = text.lower()                                      # lowercase
    text = re.sub(r"[^a-z\s]", " ", text)                    # drop punctuation/digits
    return " ".join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)  # remove stop-words

def detect_urgency(text):
    return "HIGH" if set(clean_text(text).split()) & URGENT_WORDS else "NORMAL"

# ---- train once when the server starts (uses ALL data; evaluation is in complaint_router.py) ----
df = pd.read_csv("complaints.csv")
df["clean"] = df["text"].apply(clean_text)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))             # unigrams + bigrams
X = vectorizer.fit_transform(df["clean"])                    # text -> TF-IDF matrix
clf = LogisticRegression(max_iter=1000).fit(X, df["department"])

feature_names = vectorizer.get_feature_names_out()          # every word/bigram the model knows

def top_keywords(vec, class_index, n=3):
    """Explainability: which words pushed the prediction most? (tfidf value x class weight)"""
    contrib = vec.toarray()[0] * clf.coef_[class_index]      # word importance for this class
    best = contrib.argsort()[::-1][:n]                       # indices of the n largest
    return [feature_names[i] for i in best if contrib[i] > 0]

def urgent_words_in(text):
    return sorted(set(clean_text(text).split()) & URGENT_WORDS)   # which urgent words were found

def route_complaint(text):
    """Return department, confidence, urgency, keywords and probabilities for every department."""
    vec = vectorizer.transform([clean_text(text)])
    probs = clf.predict_proba(vec)[0]
    ranking = sorted(zip(clf.classes_, probs), key=lambda p: -p[1])   # best first
    best_idx = list(clf.classes_).index(ranking[0][0])
    return {"department": ranking[0][0],
            "confidence": round(float(ranking[0][1]) * 100, 1),
            "low_confidence": bool(ranking[0][1] < 0.30),     # warn the user if the model is unsure
            "urgency": detect_urgency(text),
            "urgent_words": urgent_words_in(text),
            "keywords": top_keywords(vec, best_idx),
            "scores": [{"dept": d, "pct": round(float(p) * 100, 1)} for d, p in ranking]}
