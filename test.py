import joblib

model = joblib.load('full_spam_classifier.joblib')
print(model.predict('You won a prize'))