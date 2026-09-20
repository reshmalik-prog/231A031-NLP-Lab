"""app.py - Flask web server. Run:  python app.py   then open http://127.0.0.1:5000"""
from flask import Flask, render_template, request, jsonify
from model import route_complaint

app = Flask(__name__)

SAMPLES = [                                                  # demo complaints served to the page
    "No hot water in the hostel bathroom",
    "Found an insect in my food at the canteen",
    "I need my hall ticket tomorrow for the exam urgently",
    "The college bus was late again today",
    "The wifi in the computer lab is very slow",
    "The library is too noisy for studying",
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/samples")                                   # page loads sample buttons from here
def samples():
    return jsonify({"samples": SAMPLES})

@app.route("/predict", methods=["POST"])
def predict():
    try:                                                     # error handling: never crash the server
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Please type a complaint."}), 400
        if len(text) > 500:
            return jsonify({"error": "Complaint too long (max 500 characters)."}), 400
        return jsonify(route_complaint(text))
    except Exception as e:
        return jsonify({"error": f"Server error: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
