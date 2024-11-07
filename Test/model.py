import os
import pandas as pd
import gensim
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim_models
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Assurez-vous que les ressources NLTK sont téléchargées
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Chargement des données
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Prétraitement du texte
def preprocess_data(titles):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('french'))  # Utilise 'french' car les articles sont en français

    # Tokenisation, suppression des stopwords et lemmatisation
    processed_docs = []
    for title in titles:
        tokens = word_tokenize(title.lower())  # Convertir en minuscule et tokeniser
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
        processed_docs.append(tokens)
    
    return processed_docs

# Créer un dictionnaire et un corpus pour le modèle LDA
def create_dictionary_and_corpus(processed_docs):
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    return dictionary, corpus

# Appliquer LDA
def apply_lda(corpus, dictionary, num_topics=5):
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    return lda_model

# Visualiser les résultats
def visualize_topics(lda_model, corpus, dictionary):
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'model/lda_visualization.html')  # Sauvegarder la visualisation en HTML dans le dossier model

# Sauvegarder le modèle et les objets associés
def save_model(lda_model, dictionary):
    if not os.path.exists('model'):  # Créer le dossier 'model' s'il n'existe pas
        os.makedirs('model')
    
    lda_model.save('model/lda_model.gensim')  # Sauvegarder le modèle LDA dans le dossier model
    with open('model/lda_dictionary.pkl', 'wb') as f:
        pickle.dump(dictionary, f)  # Sauvegarder le dictionnaire dans le dossier model

def main():
    # Chemin vers le fichier CSV contenant les titres nettoyés
    filepath = 'data/preprocessed_articles.csv'
    
    # Charger les données
    df = load_data(filepath)
    
    # Prétraiter les titres
    processed_docs = preprocess_data(df['cleaned_title'])  # Utiliser 'cleaned_title' pour le traitement
    
    # Créer un dictionnaire et un corpus
    dictionary, corpus = create_dictionary_and_corpus(processed_docs)
    
    # Appliquer LDA
    lda_model = apply_lda(corpus, dictionary, num_topics=5)
    
    # Visualiser les sujets
    visualize_topics(lda_model, corpus, dictionary)
    
    # Sauvegarder le modèle et le dictionnaire
    save_model(lda_model, dictionary)
    
    print("Modèle LDA et visualisation sauvegardés avec succès dans le dossier 'model'!")

if __name__ == '__main__':
    main()
