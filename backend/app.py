from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data if not already
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def transform_text(text):
    tokens = nltk.word_tokenize(text.lower())
    main_text = [
        ps.stem(word)
        for word in tokens
        if word.isalnum() and word not in stop_words
    ]
    return ' '.join(main_text)

app = Flask(__name__)
CORS(app)

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'full_spam_classifier.joblib'))
saved = joblib.load(model_path)
classifier = saved['model']
vectorizer = saved['vectorizer']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    try:
        transformed_text = transform_text(text)

        X = vectorizer.transform([transformed_text])
        prediction = classifier.predict(X)[0]
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
