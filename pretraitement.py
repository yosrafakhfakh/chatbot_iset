import string
import re
import nltk
import unidecode

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# 🔧 Chemin pour accéder aux données nltk déjà téléchargées
nltk.data.path.append('/opt/render/nltk_data')

# Chargement différé de spaCy
SPACY_AVAILABLE = False
nlp_fr = None
nlp_en = None

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    pass


def preprocess(text, lang='fr', use_lemmatization=False):
    global nlp_fr, nlp_en

    # 1. Minuscule
    text = text.lower()

    # 2. Suppression des accents
    text = unidecode.unidecode(text)

    # 3. Suppression des élisions (l', d', etc.)
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

    # 9. Lemmatisation si activée
    if use_lemmatization and SPACY_AVAILABLE:
        if lang == 'en' and nlp_en is None:
            nlp_en = spacy.load("en_core_web_sm")
        elif lang == 'fr' and nlp_fr is None:
            nlp_fr = spacy.load("fr_core_news_sm")

        doc = nlp_en(" ".join(tokens)) if lang == 'en' else nlp_fr(" ".join(tokens))
        tokens = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
    else:
        # 10. Sinon stemming
        stemmer = SnowballStemmer('english' if lang == 'en' else 'french')
        tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)
