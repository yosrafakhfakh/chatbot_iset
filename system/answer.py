from langdetect import detect
from .loader import load_dataset
from .vectorizer import create_vectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pretraitement import preprocess

# Chargement & préparation
data = load_dataset()
questions_fr, answers_fr, questions_en, answers_en = [], [], [], []

for category in data['qa_categories'].values():
    for item in category['questions']:
        if 'fr' in item['question']:
            questions_fr.append(item['question']['fr'])
            answers_fr.append(item['reponse']['fr'])
        if 'en' in item['question']:
            questions_en.append(item['question']['en'])
            answers_en.append(item['reponse']['en'])

vectorizer_fr, X_fr = create_vectorizer(questions_fr, 'fr')
vectorizer_en, X_en = create_vectorizer(questions_en, 'en')

def format_answer(answer):
    if isinstance(answer, str):
        return answer.strip()
    if isinstance(answer, dict):
        output = ""
        for key, value in answer.items():
            if isinstance(value, list):
                output += f"<strong>{key.capitalize()}:</strong><ul>"
                for item in value:
                    output += f"<li>{item}</li>"
                output += "</ul>"
            elif isinstance(value, dict):
                output += f"<strong>{key.capitalize()}:</strong><br>" + format_answer(value) + "<br>"
            else:
                output += f"<strong>{key.capitalize()}:</strong> {value}<br>"
        return output.strip()
    elif isinstance(answer, list):
        return "<ul>" + "".join([f"<li>{format_answer(a)}</li>" for a in answer]) + "</ul>"
    return str(answer)

def get_answer(user_input):
    try:
        lang = detect(user_input)
        if lang not in ['fr', 'en']:
            lang = 'fr'
        user_input_clean = preprocess(user_input, lang=lang)
        if lang == 'fr':
            vec = vectorizer_fr.transform([user_input_clean])
            sim = cosine_similarity(vec, X_fr)
            idx = sim.argmax()
            return format_answer(answers_fr[idx]) if sim[0][idx] > 0.3 else "Désolé, je n'ai pas compris votre question. Veuillez reformuler."
        else:
            vec = vectorizer_en.transform([user_input_clean])
            sim = cosine_similarity(vec, X_en)
            idx = sim.argmax()
            return format_answer(answers_en[idx]) if sim[0][idx] > 0.3 else "Sorry, I didn't understand your question. Please try rephrasing."
    except Exception as e:
        print(f"Erreur get_answer: {e}")
        return "Une erreur est survenue."
