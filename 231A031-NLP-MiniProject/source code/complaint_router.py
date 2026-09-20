"""
Campus Complaint Router  -  NLP Mini Project
Task : Read a student's complaint (free text) -> predict the department
       (Hostel / Library / Canteen / Exam / Transport / IT) and flag urgency.
Method: Text cleaning -> TF-IDF features -> Logistic Regression (vs Naive Bayes)
        + a rule-based urgency detector.
"""

# ---------- 1. IMPORTS ----------
import re                                   # regular expressions for text cleaning
import pandas as pd                         # to load and handle the dataset table
import matplotlib.pyplot as plt             # to draw the confusion matrix
from sklearn.model_selection import train_test_split          # split data into train/test
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS  # text -> numbers
from sklearn.linear_model import LogisticRegression           # main classifier
from sklearn.naive_bayes import MultinomialNB                 # baseline classifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# ---------- 2. LOAD DATA ----------
df = pd.read_csv("complaints.csv")          # columns: text, department
print("Dataset size:", df.shape)
print(df["department"].value_counts(), "\n")

# ---------- 3. PREPROCESSING ----------
def clean_text(text):
    """lowercase -> keep only letters -> remove stop-words (the, is, a ...)"""
    text = text.lower()                                  # 'Wifi' and 'wifi' become the same word
    text = re.sub(r"[^a-z\s]", " ", text)                # remove punctuation and digits
    words = text.split()                                 # tokenization: split into words
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]   # drop useless common words
    return " ".join(words)

df["clean"] = df["text"].apply(clean_text)   # apply the function on every complaint

# ---------- 4. TRAIN / TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["department"],
    test_size=0.25,                          # 25% of data kept for testing
    random_state=42,                         # fixed seed -> same result every run
    stratify=df["department"])               # keep class balance in both parts

# ---------- 5. FEATURE EXTRACTION (TF-IDF) ----------
# TF-IDF gives a high score to words that are frequent in one complaint but rare overall
# (e.g. 'wifi', 'bus'), and a low score to words that appear everywhere.
vectorizer = TfidfVectorizer(ngram_range=(1, 2))         # unigrams + bigrams ('hot water')
X_train_vec = vectorizer.fit_transform(X_train)          # learn vocabulary from TRAIN only
X_test_vec = vectorizer.transform(X_test)                # reuse same vocabulary on TEST

# ---------- 6. TRAIN TWO MODELS ----------
models = {
    "Naive Bayes (baseline)": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
}
results = {}
for name, model in models.items():
    model.fit(X_train_vec, y_train)                      # learn from training data
    pred = model.predict(X_test_vec)                     # predict on unseen test data
    results[name] = accuracy_score(y_test, pred)
    print(f"{name}: accuracy = {results[name]:.2f}")

best_model = models["Logistic Regression"]               # use LR as the final model

# ---------- 7. DETAILED EVALUATION ----------
y_pred = best_model.predict(X_test_vec)
print("\nClassification report (Logistic Regression):")
print(classification_report(y_test, y_pred, zero_division=0))   # precision, recall, F1

cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
ConfusionMatrixDisplay(cm, display_labels=best_model.classes_).plot(cmap="Blues")
plt.title("Confusion Matrix - Complaint Router")
plt.tight_layout()
plt.savefig("confusion_matrix.png")                      # saved for your presentation slide
plt.close()

# ---------- 8. URGENCY DETECTOR (rule based) ----------
URGENT_WORDS = {"urgent", "immediately", "sick", "fire", "unsafe", "danger", "accident",
                "tomorrow", "today", "leakage", "broke", "blocked", "insect", "rash"}

def detect_urgency(text):
    words = set(clean_text(text).split())
    return "HIGH" if words & URGENT_WORDS else "NORMAL"  # any overlap with urgent words -> HIGH

# ---------- 9. PREDICTION FUNCTION ----------
def route_complaint(text):
    vec = vectorizer.transform([clean_text(text)])       # same preprocessing as training
    probs = best_model.predict_proba(vec)[0]             # probability for each department
    idx = probs.argmax()                                 # index of the most likely one
    return best_model.classes_[idx], probs[idx], detect_urgency(text)

# ---------- 10. DEMO ----------
if __name__ == "__main__":
    samples = [
        "The wifi in my hostel is not working",
        "I need the hall ticket tomorrow for my exam urgently",
        "Found an insect in my food",
        "The bus was late again today",
    ]
    print("Demo predictions:")
    for s in samples:
        dept, conf, urg = route_complaint(s)
        print(f"  '{s}'\n   -> Dept: {dept} ({conf:.0%}) | Urgency: {urg}")

    print("\nType your own complaint (or 'quit'):")
    while True:
        text = input("> ").strip()
        if text.lower() == "quit":
            break
        if text:
            dept, conf, urg = route_complaint(text)
            print(f"   Route to: {dept} ({conf:.0%} sure) | Urgency: {urg}")
