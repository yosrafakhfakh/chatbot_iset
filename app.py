from flask import Flask, request, jsonify, render_template
from system.answer import get_answer
from system.feedback import save_feedback_to_github
from dotenv import load_dotenv
from datetime import datetime

app = Flask(__name__)
load_dotenv()

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

        feedback['timestamp'] = datetime.now().isoformat()
        success = save_feedback_to_github(feedback)
        return jsonify({'status': 'Merci pour votre avis !' if success else 'Erreur lors de l\'enregistrement'}), 200
    except Exception as e:
        print(f"Erreur dans /rate: {e}")
        return jsonify({'status': 'Erreur serveur'}), 500

if __name__ == '__main__':
    app.run(debug=True)