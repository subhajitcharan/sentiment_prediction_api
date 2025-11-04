from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('stopwords')
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
encoder = joblib.load('encoder.pkl')

ps = PorterStemmer()
all_stopwords = stopwords.words('english')
if 'not' in all_stopwords:
    all_stopwords.remove('not')

def clean_text(text: str) -> str:
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = [ps.stem(word) for word in text.split() if word not in all_stopwords]
    return ' '.join(words)

app = FastAPI(title="Twitter Sentiment Analysis API")

class Tweet(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Twitter Sentiment Analysis API. Go To /docs to test the api"}

@app.post("/predict")
def predict(tweet: Tweet):
    cleaned = clean_text(tweet.text)
    X = vectorizer.transform([cleaned]).toarray()
    y_pred = model.predict(X)
    sentiment = encoder.inverse_transform(y_pred)[0]
    return {"sentiment": sentiment}
