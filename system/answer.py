from langdetect import detect
from .loader import load_dataset
from .vectorizer import create_embeddings
from pretraitement import preprocess
from sentence_transformers.util import cos_sim

# Charger les données
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

embeddings_fr = create_embeddings(questions_fr, 'fr')
embeddings_en = create_embeddings(questions_en, 'en')

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

        cleaned_input = preprocess(user_input, lang)
        input_embedding = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2").encode(cleaned_input, convert_to_tensor=True)

        if lang == 'fr':
            sim = cos_sim(input_embedding, embeddings_fr)
            idx = sim.argmax()
            return format_answer(answers_fr[idx]) if sim[0][idx] > 0.3 else "Désolé, je n'ai pas compris votre question. Veuillez reformuler."
        else:
            sim = cos_sim(input_embedding, embeddings_en)
            idx = sim.argmax()
            return format_answer(answers_en[idx]) if sim[0][idx] > 0.3 else "Sorry, I didn't understand your question. Please try rephrasing."
    except Exception as e:
        print(f"Erreur get_answer: {e}")
        return "Une erreur est survenue."
