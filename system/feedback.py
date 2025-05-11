import os
import json
import base64
import requests
from datetime import datetime
from config import GITHUB_TOKEN, REPO_NAME, REPO_OWNER, FEEDBACK_FILE

def save_feedback_to_github(feedback_data):
    try:
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }

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

        feedback_data['timestamp'] = datetime.now().isoformat()
        existing_data.append(feedback_data)

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
        print(f"Erreur feedback: {e}")
        return False
