import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score
import joblib


# Step 1: Load and preprocess data

data = pd.read_csv('CSV/spam.csv', encoding='latin1')
data = data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
data.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
data['target'] = LabelEncoder().fit_transform(data['target'])
data = data.drop_duplicates(keep='first')


nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    tokens = nltk.word_tokenize(text.lower())
    main_text = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(main_text)

data['transformed_text'] = data['text'].apply(transform_text)

# Step 2: Feature extraction

tfid = TfidfVectorizer(max_features=3000)
X = tfid.fit_transform(data['transformed_text']).toarray()
y = data['target'].values


# Step 3: Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2)

# Step 4: Train model
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = mnb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))

# Step 6: Wrap into a single classifier class
class SpamClassifier:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def transform_text(self, text):
        tokens = nltk.word_tokenize(text.lower())
        main_text = [self.ps.stem(word) for word in tokens if word.isalnum() and word not in self.stop_words]
        return ' '.join(main_text)

    def predict(self, text):
        transformed = self.transform_text(text)
        vectorized = self.vectorizer.transform([transformed]).toarray()
        prediction = self.model.predict(vectorized)
        return "Spam" if prediction[0] == 1 else "Not Spam"

# Step 7: Save model + vectorizer in one file
classifier = SpamClassifier(mnb, tfid)
joblib.dump(classifier, 'full_spam_classifier.joblib')


