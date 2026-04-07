from flask import Flask, render_template, request
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Label decoder
label_map = {
    0: "Normal",
    1: "Anxiety",
    2: "Depression",
    3: "Bipolar",
    4: "Personality Disorder",
    5: "Stress",
    6: "Suicidal"
}

# Explanation for each label
explanation_map = {
    "Normal": "No strong indicators of mental health concerns.",
    "Anxiety": "Excessive worry, restlessness, difficulty concentrating.",
    "Depression": "Persistent sadness, low energy, loss of interest.",
    "Bipolar": "Extreme mood swings from high to low.",
    "Personality Disorder": "Distorted thinking, impulsivity, difficulty in relationships.",
    "Stress": "Feeling overwhelmed, irritable, or tense.",
    "Suicidal": "Dangerous thoughts or ideation. Please seek immediate help."
}

# Flask app
app = Flask(__name__)

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned]).toarray()

    predicted_label = model.predict(vectorized)[0]
    prediction = label_map[predicted_label]
    explanation = explanation_map.get(prediction, "No explanation available.")

    return render_template('index.html', user_input=user_input, prediction=prediction, explanation=explanation)

if __name__ == "__main__":
    app.run(debug=True)
