import json

def load_dataset(path='qa_iset_dataset.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
