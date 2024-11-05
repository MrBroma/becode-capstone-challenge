import streamlit as st
import pyLDAvis
import pyLDAvis.gensim_models
from gensim import models
import pickle

# Charger le modèle, le dictionnaire et le corpus
lda_model = models.LdaModel.load('lda_model.gensim')
with open('lda_dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)
with open('lda_corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)

st.title("Topic Modeling Dashboard")

# Générer la visualisation
st.write("Visualisation des sujets extraits avec LDA")
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis_html = pyLDAvis.prepared_data_to_html(vis)
st.components.v1.html(pyLDAvis_html, width=1300, height=800)
