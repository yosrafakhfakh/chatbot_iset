#!/usr/bin/env bash

# Télécharger les stopwords NLTK
python -m nltk.downloader stopwords -d /opt/render/nltk_data

# Télécharger les modèles spaCy
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm

# Installer les dépendances Python
pip install -r requirements.txt