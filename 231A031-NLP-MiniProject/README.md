# Campus Complaint Router: NLP Text Classification with a Flask Web App

## Project Overview
This project performs **text classification** on student complaints using Natural Language Processing. A complaint typed in plain English is automatically routed to the correct college department (Hostel, Library, Canteen, Exam, Transport or IT), flagged as HIGH or NORMAL urgency, and explained with the words that influenced the decision. The model is served through a Flask web application.

## Problem Statement
Colleges receive many complaints that staff must read and forward manually. This is slow, error-prone, and urgent issues (e.g. food poisoning, water leakage) can get buried. This project automates the routing and prioritisation step.

## Dataset Details

* Name: Campus Complaints Dataset
* Source: Synthetic dataset created for this project with AI assistance (no real student data)
* Type: Short English text complaints with department labels
* Size: 96 complaints, 16 per department (balanced across 6 classes)
* File: `Dataset/complaints.csv`

## Data Preprocessing
The text was cleaned using the following steps:

* Converted all text to lowercase
* Removed punctuation and digits using regular expressions
* Tokenized text into words
* Removed English stop-words (scikit-learn list)

Split: 75% training (72 samples) and 25% testing (24 samples), stratified by department with `random_state=42`.

## Feature Extraction (TF-IDF)

* Converted cleaned text into numeric vectors using `TfidfVectorizer`
* Used unigrams and bigrams (`ngram_range=(1, 2)`) to capture phrases such as "hot water"
* Vectorizer fitted on training data only to avoid data leakage

## Model Implementation

* **Naive Bayes (Multinomial):** baseline model
* **Logistic Regression:** final model, chosen because it gives probabilities and learned word weights for explainability
* Both models trained with scikit-learn and evaluated on unseen test data

## Results

| Model | Test Accuracy |
|---|---|
| Multinomial Naive Bayes | 92% |
| Logistic Regression | 88% |

* Evaluation: accuracy, precision, recall, F1-score and confusion matrix
* Output: `Results/confusion_matrix.png`
* Observation: The few errors involve IT complaints being confused with Exam and Transport complaints. The test set has only 24 samples, so the gap between the two models is not conclusive.

## Urgency Detection

* Rule-based keyword matching (e.g. urgent, insect, sick, tomorrow, leakage)
* Returns HIGH if any urgent keyword is found, otherwise NORMAL
* This part is not machine learning

## Web Application Features

* Predicts department with confidence score
* Bar chart of scores for all departments
* Highlights urgent words inside the complaint
* Shows the top words that influenced the prediction (TF-IDF value x learned weight)
* Low-confidence warning when the top score is below 30%
* Session history table with CSV download
* Input validation and error handling

## Key Insights & Findings

* Topic words such as "wifi", "bus" and "hall ticket" are strong indicators for their departments
* TF-IDF with a linear model works well on small text datasets
* Overlapping topics (e.g. "wifi in the hostel") reduce confidence
* Bag-of-words ignores word order and context
* The dataset is small and synthetic, so real-world performance may be lower

## AI Ethics & Responsible Usage

* The dataset is synthetic and was created with AI assistance; it contains no personal or sensitive information
* Code was developed with AI assistance and then reviewed, tested and understood by the author
* This project is developed strictly for academic purposes
* Predictions should not be used to make real decisions about students without human review

## How to Run

```
cd "Source Code"
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:5000 in your browser.

To see the accuracy report and confusion matrix:
```
python complaint_router.py
```

## Repository Structure

```
Campus-Complaint-Router-NLP
│
├── Dataset/
│   └── complaints.csv
│
├── Results/
│   ├── confusion_matrix.png
│   └── app_screenshot.png
│
├── Source Code/
│   ├── app.py
│   ├── model.py
│   ├── complaint_router.py
│   ├── complaints.csv
│   ├── requirements.txt
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── script.js
│
├── AI_Ethics_Declaration_Page.pdf
└── README.md
```

## Tools & Technologies Used

* Python
* Pandas
* scikit-learn
* Matplotlib
* Flask
* HTML, CSS, JavaScript
* Chart.js
* GitHub

## Limitations & Future Work

* Small dataset; needs real, larger data and cross-validation
* English only; add Hindi and Marathi support
* Urgency is rule-based; train a separate urgency classifier
* Try transformer models such as BERT for better context understanding

## Conclusion
This project demonstrates how classical NLP techniques (TF-IDF and Logistic Regression) can automate the routing of text complaints. While effective on this small dataset, the system should be extended with real data and human review before any practical use.

## Contact
For collaboration or queries, feel free to connect.
