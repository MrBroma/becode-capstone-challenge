import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re

# Téléchargement des ressources nécessaires
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialisation des objets nécessaires
stop_words = set(stopwords.words('french'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Convertir en minuscules
    text = text.lower()
    
    # Supprimer la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Supprimer les nombres
    text = re.sub(r'\d+', '', text)
    
    # Tokenisation
    words = word_tokenize(text)
    
    # Suppression des stopwords et lemmatisation
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

def main():
    # Charger les données
    input_path = 'data/rtbf_articles.csv'
    output_path = 'data/preprocessed_articles.csv'
    
    df = pd.read_csv(input_path)
    
    # Appliquer le nettoyage au titre
    df['cleaned_title'] = df['title'].apply(clean_text)
    
    # Sauvegarder les données prétraitées
    df.to_csv(output_path, index=False)
    print(f"Data preprocessed and saved to {output_path}")

if __name__ == "__main__":
    main()
