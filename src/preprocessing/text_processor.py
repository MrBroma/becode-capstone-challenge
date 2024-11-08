import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import spacy
from spacy.tokens import Doc
import pickle
import gzip
import os
from typing import List, Dict
from collections import Counter
from tqdm import tqdm
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop_words

class AdvancedTopicModeler:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
        
        print("Loading models...")
        self.nlp = spacy.load('fr_core_news_lg')
        if 'sentencizer' not in self.nlp.pipe_names:
            self.nlp.add_pipe('sentencizer')
        
        self.sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        
        # Initialisation des attributs
        self.topic_model = None
        self.df = None
        self.embeddings = None
        self.clusters = None
        self.semantic_index = None
        
    def load_data(self):
        print("Loading data...")
        self.df = pd.read_csv(self.input_file)
        self.df['text'] = self.df['title'] + ' ' + self.df['summary'].fillna('')
        
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'], utc=True)
        
        return self
        
    def extract_semantic_info(self, doc: Doc) -> Dict:
        return {
            'entities': [
                {
                    'text': ent.text,
                    'label': ent.label_,
                    'type': self.categorize_entity(ent)
                }
                for ent in doc.ents
            ],
            'key_phrases': self.extract_key_phrases(doc),
            'main_subjects': self.extract_main_subjects(doc),
            'themes': self.extract_themes(doc),
            'sentiment': self.analyze_sentiment(doc)
        }
    
    def categorize_entity(self, ent) -> str:
        category_mapping = {
            'PER': 'PERSON',
            'ORG': 'ORGANIZATION',
            'LOC': 'LOCATION',
            'GPE': 'LOCATION',
            'EVENT': 'EVENT',
            'DATE': 'TEMPORAL',
            'TIME': 'TEMPORAL',
            'PRODUCT': 'PRODUCT',
            'WORK_OF_ART': 'CREATION'
        }
        return category_mapping.get(ent.label_, 'OTHER')
    
    def extract_key_phrases(self, doc: Doc) -> List[str]:
        key_phrases = []
        for chunk in doc.noun_chunks:
            if (not chunk.root.is_stop and
                len(chunk.text.split()) <= 4 and
                chunk.root.pos_ in ['NOUN', 'PROPN']):
                key_phrases.append(chunk.text.lower())
        return list(set(key_phrases))
    
    def extract_main_subjects(self, doc: Doc) -> List[str]:
        subjects = []
        for token in doc:
            if (token.dep_ in ['nsubj', 'nsubjpass'] and 
                token.pos_ in ['NOUN', 'PROPN']):
                subjects.append(token.text.lower())
        return list(set(subjects))
    
    def extract_themes(self, doc: Doc) -> List[str]:
        theme_words = [
            token.lemma_.lower() for token in doc
            if token.pos_ in ['NOUN', 'VERB', 'ADJ'] 
            and not token.is_stop
            and len(token.text) > 3
        ]
        return [word for word, _ in Counter(theme_words).most_common(5)]
    
    def analyze_sentiment(self, doc: Doc) -> Dict:
        """Analyse simple du sentiment"""
        positive_words = sum(1 for token in doc if token.sentiment > 0)
        negative_words = sum(1 for token in doc if token.sentiment < 0)
        total_words = len([token for token in doc if token.sentiment != 0])
        
        if total_words == 0:
            return {'score': 0, 'label': 'neutral'}
            
        score = (positive_words - negative_words) / total_words
        
        if score > 0.1:
            label = 'positive'
        elif score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
            
        return {'score': score, 'label': label}
    
    def prepare_data(self):
        print("Preprocessing and semantic analysis...")
        processed_docs = list(tqdm(self.nlp.pipe(self.df['text']), total=len(self.df)))
        semantic_info = [self.extract_semantic_info(doc) for doc in processed_docs]
        
        self.df['semantic_info'] = semantic_info
        self.df['processed_text'] = [
            ' '.join([token.lemma_.lower() for token in doc
                    if not token.is_stop and not token.is_punct and token.text.lower() not in fr_stop_words])
            for doc in processed_docs
        ]
        
        # Extraction des caractéristiques
        self.df['entities'] = [info['entities'] for info in semantic_info]
        self.df['key_phrases'] = [info['key_phrases'] for info in semantic_info]
        self.df['main_subjects'] = [info['main_subjects'] for info in semantic_info]
        self.df['themes'] = [info['themes'] for info in semantic_info]
        self.df['sentiment'] = [info['sentiment'] for info in semantic_info]
        
        print("Generating embeddings...")
        text_for_embedding = [
            f"{text} {' '.join(kp)} {' '.join(themes)}"
            for text, kp, themes in zip(
                self.df['processed_text'],
                self.df['key_phrases'],
                self.df['themes']
            )
        ]
        
        self.embeddings = np.array(self.sentence_model.encode(
            text_for_embedding,
            batch_size=32,
            show_progress_bar=True
        ))
        
        return self

    def perform_clustering(self, n_clusters: int = 10):
            print("Clustering articles...")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.clusters = kmeans.fit_predict(self.embeddings)
            self.df['cluster'] = self.clusters
            
            # Identifier les articles représentatifs
            cluster_centers = kmeans.cluster_centers_
            self.df['is_representative'] = False
            
            for cluster in range(n_clusters):
                cluster_mask = self.df['cluster'] == cluster
                if cluster_mask.any():
                    cluster_indices = self.df[cluster_mask].index
                    cluster_docs = self.embeddings[cluster_mask]
                    
                    distances = np.linalg.norm(cluster_docs - cluster_centers[cluster], axis=1)
                    closest_idx_in_subset = np.argmin(distances)
                    representative_idx = cluster_indices[closest_idx_in_subset]
                    
                    self.df.loc[representative_idx, 'is_representative'] = True
                    
                    cluster_articles = self.df[cluster_mask]
                    
                    all_entities = []
                    for entities in cluster_articles['entities']:
                        all_entities.extend([
                            entity['text'] 
                            for entity in entities
                            if entity['type'] in ['PERSON', 'ORGANIZATION', 'LOCATION']
                        ])
                    common_entities = Counter(all_entities).most_common(3)
                    
                    all_themes = []
                    for themes in cluster_articles['themes']:
                        all_themes.extend(themes)
                    main_themes = Counter(all_themes).most_common(3)
                    
                    cluster_summary = {
                        'themes': [theme for theme, _ in main_themes],
                        'entities': [entity for entity, _ in common_entities],
                        'size': len(cluster_articles),
                        'period': f"{cluster_articles['date'].min():%Y-%m-%d} to {cluster_articles['date'].max():%Y-%m-%d}"
                    }
                    
                    self.df.loc[cluster_mask, 'cluster_summary'] = str(cluster_summary)
            
            return self
        
    def train_model(self, n_topics: int = 15):
        """Entraînement du modèle avec BERTopic"""
        print("Training BERTopic model...")

        self.topic_model = BERTopic(
            language="french",
            nr_topics=n_topics,
            min_topic_size=5,
            embedding_model=None,
            verbose=True
        )

        enriched_docs = [
            f"{text} {' '.join(kp)} {' '.join(themes)}"
            for text, kp, themes in zip(
                self.df['processed_text'],
                self.df['key_phrases'],
                self.df['themes']
            )
        ]

        topics, probs = self.topic_model.fit_transform(
            enriched_docs,
            embeddings=self.embeddings
        )

        if probs.ndim == 1:
            probs = probs[:, None]

        self.df['topic'] = topics
        self.df['topic_probability'] = probs.max(axis=1)

        self.topic_names = {}
        for topic_id in set(topics):
            if topic_id != -1:
                topic_docs = self.df[self.df['topic'] == topic_id]

                topic_keywords = [word for word, _ in self.topic_model.get_topic(topic_id)[:3]]

                top_entities = [
                    ent['text'] for ents in topic_docs['entities']
                    for ent in ents if ent['type'] in ['ORGANIZATION', 'LOCATION', 'PERSON']
                ]

                if top_entities:
                    most_common_entity = Counter(top_entities).most_common(1)[0][0]
                    topic_keywords.append(most_common_entity)

                self.topic_names[topic_id] = " - ".join(topic_keywords[:3])

                topic_summary = {
                    'keywords': topic_keywords,
                    'entities': Counter(top_entities).most_common(3),
                    'themes': Counter([
                        theme for themes in topic_docs['themes']
                        for theme in themes
                    ]).most_common(3),
                    'sentiment': Counter(
                        topic_docs['sentiment'].apply(lambda x: x['label'])
                    ).most_common(1)[0][0]
                }

                self.df.loc[self.df['topic'] == topic_id, 'topic_summary'] = str(topic_summary)

        self.df['topic_name'] = self.df['topic'].map(
            lambda x: self.topic_names.get(x, f"Topic {x}")
        )

        return self
    
    def create_semantic_index(self):
        """Crée un index sémantique pour la recherche avancée"""
        print("Création de l'index sémantique...")
        
        self.semantic_index = NearestNeighbors(
            n_neighbors=min(50, len(self.df)),
            metric='cosine'
        )
        self.semantic_index.fit(self.embeddings)
        
        return self
    
    def save_results(self):
        """Sauvegarde des résultats enrichis"""
        print("Sauvegarde des résultats...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Préparation des données pour la sauvegarde
        output_df = pd.DataFrame({
            'title': self.df['title'],
            'summary': self.df['summary'],
            'topic': self.df['topic'],
            'topic_name': self.df['topic_name'],
            'topic_probability': self.df['topic_probability'],
            'topic_summary': self.df.get('topic_summary', ''),
            'cluster': self.df['cluster'],
            'cluster_summary': self.df.get('cluster_summary', ''),
            'is_representative': self.df['is_representative'],
            'entities': self.df['entities'],
            'key_phrases': self.df['key_phrases'],
            'main_subjects': self.df['main_subjects'],
            'themes': self.df['themes'],
            'sentiment': self.df['sentiment'],
            'link': self.df['link'],
            'category': self.df['category'],
            'date': self.df['date']
        })
        
        output_df.to_csv(
            os.path.join(self.output_dir, 'processed_articles.csv.gz'),
            index=False,
            compression='gzip'
        )
        
        np.savez_compressed(
            os.path.join(self.output_dir, 'embeddings.npz'),
            embeddings=self.embeddings
        )
        
        with gzip.open(os.path.join(self.output_dir, 'semantic_index.pkl.gz'), 'wb') as f:
            pickle.dump(self.semantic_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if hasattr(self.topic_model, 'embedding_model'):
            self.topic_model.embedding_model = None
            
        with gzip.open(os.path.join(self.output_dir, 'topic_model.pkl.gz'), 'wb') as f:
            pickle.dump(self.topic_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return self
    
    def process_all(self, n_topics: int = 15, n_clusters: int = 10):
        (self.load_data()
             .prepare_data()
             .perform_clustering(n_clusters)
             .train_model(n_topics)
             .create_semantic_index()
             .save_results())
        return self

def main():
    input_file = "data/raw/rtbf_articles.csv"
    output_dir = "data/processed"
    n_topics = 15
    n_clusters = 10
    
    modeler = AdvancedTopicModeler(input_file, output_dir)
    modeler.load_data()
    modeler.prepare_data()
    modeler.perform_clustering(n_clusters)
    modeler.train_model(n_topics)
    modeler.create_semantic_index()
    modeler.save_results()

if __name__ == "__main__":
    main()

