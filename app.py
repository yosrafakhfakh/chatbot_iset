from flask import Flask, request, jsonify, render_template
import json
import os
import requests
import base64
from datetime import datetime
from pretraitement import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from dotenv import load_dotenv


app = Flask(__name__)

load_dotenv()
# Configuration GitHub (récupérée depuis les variables d'environnement)
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_OWNER = "yosrafakhfakh"  # Remplacez par votre nom d'utilisateur GitHub
REPO_NAME = "mini-chatbot"    # Remplacez par votre nom de dépôt
FEEDBACK_FILE = "ratings.json"

# Chargement du dataset
with open('qa_iset_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Séparation des questions/réponses par langue
questions_fr, answers_fr = [], []
questions_en, answers_en = [], []

for category in data['qa_categories'].values():
    for item in category['questions']:
        if 'fr' in item['question']:
            questions_fr.append(item['question']['fr'])
            answers_fr.append(item['reponse']['fr'])
        if 'en' in item['question']:
            questions_en.append(item['question']['en'])
            answers_en.append(item['reponse']['en'])

# Vectorisation TF-IDF
vectorizer_fr = TfidfVectorizer()
X_fr = vectorizer_fr.fit_transform([preprocess(q, lang='fr') for q in questions_fr])

vectorizer_en = TfidfVectorizer()
X_en = vectorizer_en.fit_transform([preprocess(q, lang='en') for q in questions_en])

def format_answer(answer):
    """Formate la réponse pour l'affichage HTML"""
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
    """Trouve la réponse la plus pertinente"""
    try:
        lang = detect(user_input)
        if lang not in ['fr', 'en']:
            lang = 'fr'

        user_input_clean = preprocess(user_input, lang=lang)
        
        if lang == 'fr':
            user_vec = vectorizer_fr.transform([user_input_clean])
            similarities = cosine_similarity(user_vec, X_fr)
            best_index = similarities.argmax()
            score = similarities[0][best_index]
            
            return format_answer(answers_fr[best_index]) if score > 0.3 else "Désolé, je n'ai pas compris votre question. Veuillez reformuler."
        else:
            user_vec = vectorizer_en.transform([user_input_clean])
            similarities = cosine_similarity(user_vec, X_en)
            best_index = similarities.argmax()
            score = similarities[0][best_index]
            
            return format_answer(answers_en[best_index]) if score > 0.3 else "Sorry, I didn't understand your question. Please try rephrasing."
    
    except Exception as e:
        print(f"Erreur dans get_answer: {e}")
        return "Une erreur est survenue. Veuillez réessayer."

def save_feedback_to_github(feedback_data):
    """Enregistre le feedback dans le dépôt GitHub"""
    try:
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Récupère le contenu actuel du fichier
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FEEDBACK_FILE}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            file_data = response.json()
            content = requests.get(file_data['download_url']).text
            existing_data = json.loads(content) if content else []
            sha = file_data['sha']
        else:
            existing_data = []
            sha = None
        
        # Ajoute le nouveau feedback
        feedback_data['timestamp'] = datetime.now().isoformat()
        existing_data.append(feedback_data)
        
        # Met à jour le fichier sur GitHub
        update_data = {
            "message": f"Feedback update {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "content": base64.b64encode(json.dumps(existing_data, indent=2).encode('utf-8')).decode('utf-8'),
            "branch": "main"
        }
        
        if sha:
            update_data["sha"] = sha
            
        response = requests.put(url, headers=headers, json=update_data)
        return response.status_code in [200, 201]
    
    except Exception as e:
        print(f"Erreur lors de l'envoi du feedback: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_input = request.json.get('question', '')
        answer = get_answer(user_input)
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Erreur dans /ask: {e}")
        return jsonify({'answer': 'Une erreur est survenue'}), 500

@app.route('/rate', methods=['POST'])
def rate():
    try:
        feedback = request.json
        if not feedback:
            return jsonify({'status': 'Aucun feedback reçu'}), 400
        
        if not GITHUB_TOKEN:
            print("Erreur : GITHUB_TOKEN non défini")
            return jsonify({'status': 'Token GitHub manquant'}), 500

        # Ajoute un timestamp au feedback
        feedback['timestamp'] = datetime.now().isoformat()

        # Enregistrer le feedback sur GitHub
        success = save_feedback_to_github(feedback)

        return jsonify({'status': 'Merci pour votre avis !' if success else 'Erreur lors de l\'enregistrement'}), 200
    except Exception as e:
        print(f"Erreur dans /rate: {e}")
        return jsonify({'status': 'Erreur serveur'}), 500


def save_feedback_to_github(feedback_data):
    """Enregistre le feedback dans le fichier GitHub"""
    try:
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }

        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FEEDBACK_FILE}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            file_data = response.json()
            content = base64.b64decode(file_data['content']).decode('utf-8')
            existing_data = json.loads(content) if content else []
            sha = file_data['sha']
        elif response.status_code == 404:
            print("Le fichier n'existe pas encore. Il sera créé.")
            existing_data = []
            sha = None
        else:
            print(f"Erreur lors de la récupération du fichier : {response.status_code} {response.text}")
            return False

        feedback_data['timestamp'] = datetime.now().isoformat()
        existing_data.append(feedback_data)

        update_data = {
            "message": f"Feedback update {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "content": base64.b64encode(json.dumps(existing_data, indent=2).encode('utf-8')).decode('utf-8'),
            "branch": "main"
        }

        if sha:
            update_data["sha"] = sha

        put_response = requests.put(url, headers=headers, json=update_data)

        print(f"PUT GitHub status: {put_response.status_code}")
        print(f"GitHub response: {put_response.text}")

        return put_response.status_code in [200, 201]

    except Exception as e:
        print(f"Erreur lors de l'envoi du feedback: {e}")
        return False


if __name__ == '__main__':
    app.run(debug=True)