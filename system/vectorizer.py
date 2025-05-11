from sklearn.feature_extraction.text import TfidfVectorizer
from pretraitement import preprocess

def create_vectorizer(questions, lang):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([preprocess(q, lang=lang) for q in questions])
    return vectorizer, X
