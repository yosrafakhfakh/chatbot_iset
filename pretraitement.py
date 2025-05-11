import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('stopwords')

def preprocess(text, lang='fr'):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()  # Remplace word_tokenize
    stop_words = stopwords.words('english' if lang == 'en' else 'french')
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    stemmer = SnowballStemmer('english' if lang == 'en' else 'french')
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)
