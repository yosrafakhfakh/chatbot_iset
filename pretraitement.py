import string
import re
import nltk
import unidecode

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

try:
    import spacy
    nlp_fr = spacy.load("fr_core_news_sm")
    nlp_en = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False

nltk.download('stopwords')

def preprocess(text, lang='fr', use_lemmatization=False):
    # 1. Minuscule
    text = text.lower()

    # 2. Suppression des accents
    text = unidecode.unidecode(text)

    # 3. Suppression des élisions (l', d', qu', etc.)
    text = re.sub(r"\b[ldjtmcqs]['’]", "", text)

    # 4. Suppression ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 5. Suppression des chiffres
    text = re.sub(r'\d+', '', text)

    # 6. Tokenisation simple
    tokens = text.split()

    # 7. Suppression des stopwords
    stop_words = stopwords.words('english' if lang == 'en' else 'french')
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # 8. Suppression des mots très courts
    tokens = [word for word in tokens if len(word) > 2]

    # 9. Lemmatisation (si activée et possible)
    if use_lemmatization and SPACY_AVAILABLE:
        doc = nlp_en(" ".join(tokens)) if lang == 'en' else nlp_fr(" ".join(tokens))
        tokens = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
    else:
        # 10. Stemming par défaut
        stemmer = SnowballStemmer('english' if lang == 'en' else 'french')
        tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)
