import spacy
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from collections import Counter

class SearchEngine:
    def __init__(self, nlp_model='fr_core_news_lg'):
        """Initialise le moteur de recherche avec des capacités NLP avancées"""
        self.nlp = spacy.load(nlp_model)
        
        # Patterns de questions améliorés
        self.question_patterns = {
            'que': ['que', "qu'est-ce", 'quoi', 'quel', 'quelle', 'quels', 'quelles'],
            'qui': ['qui', 'quelle personne', 'quelles personnes'],
            'où': ['où', 'quel endroit', 'quel lieu', 'quelle ville', 'quel pays'],
            'quand': ['quand', 'quelle date', 'quel jour', 'quelle période'],
            'comment': ['comment', 'de quelle manière', 'par quel moyen'],
            'pourquoi': ['pourquoi', 'pour quelle raison', 'à cause de quoi']
        }
        
        # Expressions temporelles étendues
        self.time_expressions = {
            "aujourd'hui": 1,
            "hier": 2,
            "cette semaine": 7,
            "la semaine dernière": 14,
            "ce mois": 30,
            "le mois dernier": 60,
            "cette année": 365,
            "récent": 7,
            "derniers jours": 7,
            "dernière semaine": 7,
            "dernier mois": 30,
            "dernières semaines": 14
        }
        
        # Dictionnaire des types d'entités
        self.entity_types = {
            'PERSON': ['personne', 'qui', 'quel'],
            'LOCATION': ['où', 'lieu', 'endroit', 'ville', 'pays'],
            'ORGANIZATION': ['organisation', 'entreprise', 'société'],
            'EVENT': ['événement', 'manifestation', 'conférence'],
            'DATE': ['quand', 'date', 'période'],
            'TOPIC': ['sujet', 'thème', 'domaine']
        }

    def parse_query(self, query: str) -> Dict:
        """Analyse une requête en langage naturel"""
        doc = self.nlp(query.lower())
        
        # Extraction des entités
        entities = {ent.label_: ent.text for ent in doc.ents}
        
        # Extraction de la période temporelle
        time_refs = {
            "aujourd'hui": 1,
            "hier": 2,
            "cette semaine": 7,
            "la semaine dernière": 14,
            "ce mois": 30,
            "le mois dernier": 60,
            "cette année": 365
        }
        
        days_filter = None
        for time_ref, days in time_refs.items():
            if time_ref in query.lower():
                days_filter = days
                break
        
        # Si aucune période n'est spécifiée, utiliser une période par défaut
        if days_filter is None:
            days_filter = 30  # Par défaut, chercher sur le dernier mois
        
        # Extraction des mots-clés principaux
        keywords = [
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct
            and token.pos_ in ['NOUN', 'PROPN', 'ADJ']
            and len(token.text) > 2
        ]
        
        # Détection du type de recherche
        search_type = self._detect_search_type(doc)
        
        return {
            'entities': entities,
            'days_filter': days_filter,
            'keywords': keywords,
            'search_type': search_type,
            'original_query': query  # Ajouter la requête originale
        }

    def _detect_search_type(self, doc) -> str:
        """Détecte le type de recherche basé sur la structure de la question"""
        question_words = {
            'qui': 'person',
            'où': 'location',
            'quand': 'temporal',
            'que': 'event',
            'quoi': 'event',
            'comment': 'explanation',
            'pourquoi': 'explanation'
        }
        
        # Vérifier les mots interrogatifs
        text = doc.text.lower()
        for word, search_type in question_words.items():
            if word in text:
                return search_type
        
        # Si aucun mot interrogatif n'est trouvé
        return 'general'

    def _detect_question_type(self, doc) -> str:
        """Détection améliorée du type de question"""
        text = doc.text.lower()
        
        # Vérification des patterns explicites
        for q_type, patterns in self.question_patterns.items():
            if any(pattern in text for pattern in patterns):
                return q_type

        # Analyse de la structure syntaxique
        for token in doc:
            # Détection des verbes d'action spécifiques
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                if token.lemma_ in ['montrer', 'afficher', 'trouver', 'chercher']:
                    return 'que'
                elif token.lemma_ in ['expliquer', 'décrire']:
                    return 'explanation'
        
        return 'general'

    def _extract_temporal_context(self, doc) -> Optional[int]:
        """Extraction avancée du contexte temporel"""
        text = doc.text.lower()
        
        # Vérification des expressions explicites
        for expr, days in self.time_expressions.items():
            if expr in text:
                return days
        
        # Analyse des entités temporelles
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                # Analyse des expressions relatives
                if any(word in ent.text.lower() for word in ['dernier', 'passé', 'précédent']):
                    return 30
                if any(word in ent.text.lower() for word in ['prochain', 'futur']):
                    return None
                
                # Tentative de conversion en date explicite
                try:
                    date = pd.to_datetime(ent.text, format='%d/%m/%Y')
                    delta = datetime.now() - date
                    return delta.days
                except:
                    pass
        
        return 7  # Valeur par défaut

    def _extract_entities(self, doc) -> Dict[str, List[Dict]]:
        """Extraction enrichie des entités"""
        entities = {'named': [], 'contextual': [], 'temporal': []}
        
        for ent in doc.ents:
            entity_data = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'context': self._get_entity_context(doc, ent)
            }
            
            if ent.label_ in ['DATE', 'TIME']:
                entities['temporal'].append(entity_data)
            else:
                entities['named'].append(entity_data)
        
        # Extraction des entités contextuelles
        for chunk in doc.noun_chunks:
            if not any(ent.text == chunk.text for ent in doc.ents):
                entities['contextual'].append({
                    'text': chunk.text,
                    'root': chunk.root.text,
                    'root_dep': chunk.root.dep_
                })
        
        return entities

    def _get_entity_context(self, doc, ent, window=3):
        """Extrait le contexte autour d'une entité"""
        start = max(0, ent.start - window)
        end = min(len(doc), ent.end + window)
        return doc[start:end].text

    def _extract_keywords(self, doc) -> List[Dict[str, float]]:
        """Extraction de mots-clés avec pondération"""
        keywords = []
        seen = set()
        
        for token in doc:
            if (not token.is_stop and not token.is_punct and 
                token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB'] and
                len(token.text) > 2 and token.lemma_ not in seen):
                
                # Calcul du score
                score = 1.0
                
                # Bonus pour les noms propres
                if token.pos_ == 'PROPN':
                    score *= 1.5
                
                # Bonus pour les mots en majuscules
                if token.text.isupper():
                    score *= 1.2
                
                # Bonus pour les mots proches des mots interrogatifs
                if self._is_near_question_word(token):
                    score *= 1.3
                
                keywords.append({
                    'text': token.lemma_,
                    'pos': token.pos_,
                    'score': score
                })
                seen.add(token.lemma_)
        
        return sorted(keywords, key=lambda x: x['score'], reverse=True)

    def _is_near_question_word(self, token, window=3):
        """Vérifie si un token est proche d'un mot interrogatif"""
        surrounding = list(token.doc[max(0, token.i - window):token.i + window + 1])
        question_words = [word for pattern in self.question_patterns.values() for word in pattern]
        return any(t.text.lower() in question_words for t in surrounding)

    def _extract_topics(self, doc) -> List[str]:
        """Extraction des sujets/thèmes de la requête"""
        topic_indicators = ['sur', 'concernant', 'à propos de', 'au sujet de']
        topics = []
        
        for chunk in doc.noun_chunks:
            for indicator in topic_indicators:
                if indicator in chunk.root.head.text.lower():
                    topics.append(chunk.text)
                    break
        
        return topics

    def _detect_intent(self, doc) -> Dict[str, float]:
        """Détection de l'intention de la requête"""
        intents = {
            'search': 0.0,  # Recherche d'information
            'compare': 0.0,  # Comparaison
            'explain': 0.0,  # Explication
            'timeline': 0.0,  # Chronologie
            'summary': 0.0   # Résumé
        }
        
        # Mots-clés associés aux intentions
        intent_keywords = {
            'search': ['chercher', 'trouver', 'rechercher', 'montrer'],
            'compare': ['comparer', 'différence', 'versus', 'entre'],
            'explain': ['expliquer', 'pourquoi', 'comment', 'raison'],
            'timeline': ['quand', 'chronologie', 'évolution', 'temporel'],
            'summary': ['résumer', 'synthèse', 'bref', 'résumé']
        }
        
        # Analyse des verbes et mots-clés
        for token in doc:
            for intent, keywords in intent_keywords.items():
                if token.lemma_ in keywords:
                    intents[intent] += 1.0
        
        # Normalisation
        total = sum(intents.values()) or 1
        return {k: v/total for k, v in intents.items()}

    def _enrich_query_data(self, base_analysis: Dict, doc) -> Dict:
        """Enrichissement des données de la requête"""
        # Ajout de métadonnées
        metadata = {
            'query_length': len(doc),
            'has_question_mark': '?' in doc.text,
            'has_temporal_reference': bool(base_analysis['temporal_context']),
            'complexity': self._calculate_query_complexity(doc)
        }
        
        # Fusion et retour des données enrichies
        return {
            **base_analysis,
            'metadata': metadata,
            'suggestions': self._generate_query_suggestions(base_analysis)
        }

    def _calculate_query_complexity(self, doc) -> float:
        """Calcule la complexité de la requête"""
        factors = {
            'length': len(doc) / 20,  # Normalisé pour ~20 mots
            'entities': len([ent for ent in doc.ents]) / 3,
            'syntax': len([token for token in doc if token.dep_ in ['nsubj', 'dobj', 'pobj']]) / 5
        }
        return sum(factors.values()) / len(factors)

    def _generate_query_suggestions(self, analysis: Dict) -> List[str]:
        """Génère des suggestions de requêtes alternatives"""
        suggestions = []
        
        # Suggestion basée sur le type de question
        if analysis['question_type'] == 'general':
            pattern = self.question_patterns.get(analysis['intent'])
            if pattern:
                suggestions.append(f"Essayez: '{pattern[0]} ...'")
        
        # Suggestion temporelle si absente
        if not analysis['temporal_context']:
            suggestions.append("Précisez la période (exemple: 'cette semaine', 'ce mois')")
        
        return suggestions

    def create_search_filters(self, query_info: Dict, df: pd.DataFrame) -> pd.Series:
        """Crée des filtres de recherche améliorés"""
        base_mask = pd.Series(True, index=df.index)
        
        # Filtre temporel
        if query_info.get('days_filter'):
            cutoff_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=query_info['days_filter'])
            base_mask &= (df['date'] > cutoff_date)
        
        # Filtre par entités
        for ent_type, ent_text in query_info['entities'].items():
            if ent_type == 'LOC':
                base_mask &= df['entities'].apply(
                    lambda x: any(
                        ent_text.lower() in ent['text'].lower()
                        for ent in x if ent['type'] == 'LOCATION'
                    )
                )
            elif ent_type == 'PER':
                base_mask &= df['entities'].apply(
                    lambda x: any(
                        ent_text.lower() in ent['text'].lower()
                        for ent in x if ent['type'] == 'PERSON'
                    )
                )
        
        # Filtre par mots-clés
        if query_info['keywords']:
            keywords_mask = pd.Series(False, index=df.index)
            for keyword in query_info['keywords']:
                keyword_text = keyword.lower()
                keyword_matches = (
                    df['title'].str.lower().str.contains(keyword_text, na=False) |
                    df['summary'].str.lower().str.contains(keyword_text, na=False)
                )
                keywords_mask |= keyword_matches
            base_mask &= keywords_mask
        
        return base_mask

    def rank_results(self, df: pd.DataFrame, query_info: Dict, mask: pd.Series) -> pd.DataFrame:
        """Range les résultats par pertinence"""
        results = df[mask].copy()
        
        if len(results) == 0:
            return results
        
        # Calcul du score de pertinence
        results['relevance'] = 0.0
        
        # Score basé sur les mots-clés
        for keyword in query_info['keywords']:
            keyword_text = keyword.lower()
            results['relevance'] += (
                results['title'].str.lower().str.contains(keyword_text, na=False).astype(float) * 2 +
                results['summary'].str.lower().str.contains(keyword_text, na=False).astype(float)
            )
        
        # Score basé sur les entités
        for ent_type, ent_text in query_info['entities'].items():
            results['relevance'] += results['entities'].apply(
                lambda x: sum(2 if ent['text'].lower() == ent_text.lower() else 1
                            for ent in x if ent['type'] == ent_type)
            )
        
        # Bonus pour la fraîcheur des articles
        if 'date' in results.columns:
            max_date = results['date'].max()
            min_date = results['date'].min()
            if max_date != min_date:
                results['relevance'] *= 1 + 0.2 * (
                    (results['date'] - min_date) / (max_date - min_date)
                )
        
        # Normalisation du score
        max_score = results['relevance'].max()
        if max_score > 0:
            results['relevance'] = (results['relevance'] / max_score) * 100
        
        # Tri par pertinence et date
        results = results.sort_values(['relevance', 'date'], ascending=[False, False])
        
        return results

    def format_search_results(self, results: pd.DataFrame, query_info: Dict) -> str:
        """Formate un résumé des résultats de recherche"""
        if len(results) == 0:
            return "Aucun article trouvé."
                
        summary_parts = []
            
        # Résumé temporel
        if 'date' in results.columns:
            date_range = f"du {results['date'].min().strftime('%d/%m/%Y')} au {results['date'].max().strftime('%d/%m/%Y')}"
            summary_parts.append(f"Articles {date_range}")
            
        # Analyse des entités par type
        entity_counts = {
            'Personnes': [],
            'Organisations': [],
            'Lieux': []
        }
        
        all_entities = []
        for ents in results['entities']:
            for ent in ents:
                if ent['type'] == 'PERSON':
                    entity_counts['Personnes'].append(ent['text'])
                elif ent['type'] == 'ORGANIZATION':
                    entity_counts['Organisations'].append(ent['text'])
                elif ent['type'] == 'LOCATION':
                    entity_counts['Lieux'].append(ent['text'])
                all_entities.append(ent['text'])
        
        # Thèmes principaux
        all_themes = [theme for themes in results['themes'] for theme in themes]
        if all_themes:
            top_themes = Counter(all_themes).most_common(3)
            summary_parts.append(f"Thèmes principaux: {', '.join(f'{theme} ({count})' for theme, count in top_themes)}")
        
        # Ajout des entités principales par type
        for entity_type, entities in entity_counts.items():
            if entities:
                top_entities = Counter(entities).most_common(3)
                summary_parts.append(f"{entity_type}: {', '.join(f'{ent} ({count})' for ent, count in top_entities)}")
        
        return " | ".join(summary_parts)

    def _map_entity_type(self, spacy_type: str) -> str:
        """Conversion des types d'entités Spacy vers les types internes"""
        mapping = {
            'PER': 'PERSON',
            'LOC': 'LOCATION',
            'GPE': 'LOCATION',
            'ORG': 'ORGANIZATION',
            'DATE': 'TEMPORAL',
            'EVENT': 'EVENT'
        }
        return mapping.get(spacy_type, spacy_type)

    def _summarize_entities(self, results: pd.DataFrame) -> str:
        """Résumé détaillé des entités trouvées"""
        entity_types = ['PERSON', 'ORGANIZATION', 'LOCATION']
        entity_summaries = []
        
        for entity_type in entity_types:
            entities = [
                ent['text'] for ents in results['entities']
                for ent in ents if ent['type'] == entity_type
            ]
            
            if entities:
                top_entities = Counter(entities).most_common(3)
                type_name = {
                    'PERSON': 'Personnes',
                    'ORGANIZATION': 'Organisations',
                    'LOCATION': 'Lieux'
                }.get(entity_type, entity_type)
                
                entities_text = ', '.join(f"{e} ({c})" for e, c in top_entities)
                entity_summaries.append(f"{type_name}: {entities_text}")
        
        return " | ".join(entity_summaries) if entity_summaries else ""

    def _generate_alternative_suggestions(self, query_info: Dict) -> List[str]:
        """Génère des suggestions alternatives en cas de résultats vides"""
        suggestions = []
        
        # Suggestion basée sur les mots-clés
        if query_info['keywords']:
            suggestions.append(
                "Essayez avec moins de mots-clés ou des termes plus généraux"
            )
        
        # Suggestion basée sur le filtre temporel
        if query_info['temporal_context']:
            suggestions.append(
                "Élargissez la période de recherche"
            )
        
        # Suggestion basée sur les entités
        if query_info['entities']['named']:
            suggestions.append(
                "Vérifiez l'orthographe des noms propres ou utilisez des termes plus généraux"
            )
        
        return suggestions if suggestions else ["Reformulez votre question différemment"]

    def _generate_refinement_suggestions(self, results: pd.DataFrame, query_info: Dict) -> List[str]:
        """Génère des suggestions pour affiner les résultats"""
        suggestions = []
        
        # Si trop de résultats
        if len(results) > 50:
            if not query_info.get('days_filter'):  # Changé de 'temporal_context' à 'days_filter'
                suggestions.append("Précisez une période temporelle pour affiner les résultats")
            
            if not query_info['entities']:
                # Suggérer les entités les plus fréquentes
                top_entities = [
                    ent['text'] for ents in results.head(10)['entities']
                    for ent in ents if ent['type'] in ['PERSON', 'ORGANIZATION', 'LOCATION']
                ]
                if top_entities:
                    most_common = Counter(top_entities).most_common(1)[0][0]
                    suggestions.append(f"Précisez votre recherche en incluant '{most_common}'")
        
        # Si peu de résultats
        elif len(results) < 5:
            suggestions.append("Essayez des termes plus généraux ou une période plus large")
            
            # Si filtres spécifiques sont appliqués
            if query_info.get('days_filter', 0) < 30:  # Changé de 'temporal_context' à 'days_filter'
                suggestions.append("Élargissez la période de recherche")
        
        # Si score de pertinence faible
        if 'relevance' in results.columns and len(results) > 0:
            avg_relevance = results['relevance'].mean()
            if avg_relevance < 30:
                suggestions.append("Utilisez des termes plus spécifiques pour obtenir des résultats plus pertinents")
        
        return suggestions