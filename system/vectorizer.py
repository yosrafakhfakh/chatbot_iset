from sentence_transformers import SentenceTransformer
from pretraitement import preprocess

model_name = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

def create_embeddings(questions, lang):
    cleaned = [preprocess(q, lang) for q in questions]
    embeddings = model.encode(cleaned, convert_to_tensor=True)
    return embeddings
