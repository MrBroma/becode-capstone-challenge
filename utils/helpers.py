from typing import Dict, List, Union
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

class DataVisualizer:
    @staticmethod
    def create_timeline_plot(df: pd.DataFrame, title: str = "Distribution temporelle") -> go.Figure:
        """Crée un graphique temporel des articles"""
        articles_per_day = df.groupby(df['date'].dt.date).size().reset_index()
        articles_per_day.columns = ['date', 'count']
        
        fig = px.line(
            articles_per_day,
            x='date',
            y='count',
            labels={'date': 'Date', 'count': 'Nombre d\'articles'},
            title=title
        )
        fig.update_layout(
            hovermode='x unified',
            showlegend=True
        )
        return fig

    @staticmethod
    def create_topic_distribution(df: pd.DataFrame) -> go.Figure:
        """Crée un graphique de distribution des topics"""
        topic_counts = df.groupby(['topic_name'])['topic'].count().reset_index()
        topic_counts.columns = ['Topic', 'Count']
        
        fig = px.bar(
            topic_counts,
            x='Topic',
            y='Count',
            title='Distribution des Topics'
        )
        fig.update_layout(
            xaxis_tickangle=45,
            height=500
        )
        return fig

class TextProcessor:
    @staticmethod
    def format_date(date: Union[str, datetime], format_type: str = 'short') -> str:
        """Formate une date selon le type demandé"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        if format_type == 'short':
            return date.strftime('%d/%m/%Y')
        elif format_type == 'long':
            return date.strftime('%d/%m/%Y %H:%M')
        elif format_type == 'relative':
            now = datetime.now()
            diff = now - date
            
            if diff.days == 0:
                if diff.seconds < 3600:
                    minutes = diff.seconds // 60
                    return f"il y a {minutes} minute{'s' if minutes > 1 else ''}"
                else:
                    hours = diff.seconds // 3600
                    return f"il y a {hours} heure{'s' if hours > 1 else ''}"
            elif diff.days == 1:
                return "hier"
            elif diff.days < 7:
                return f"il y a {diff.days} jours"
            elif diff.days < 30:
                weeks = diff.days // 7
                return f"il y a {weeks} semaine{'s' if weeks > 1 else ''}"
            else:
                return date.strftime('%d/%m/%Y')
    
    @staticmethod
    def highlight_text(text: str, keywords: List[str]) -> str:
        """Met en évidence les mots-clés dans un texte"""
        highlighted = text
        for keyword in keywords:
            if not keyword.strip():
                continue
            highlighted = highlighted.replace(
                keyword,
                f"**{keyword}**"
            )
        return highlighted

class ResultsFormatter:
    @staticmethod
    def format_entities(entities: List[Dict]) -> str:
        """Formate les entités pour l'affichage"""
        if not entities:
            return ""
            
        formatted_entities = []
        for ent in entities[:3]:  # Limite aux 3 premières entités
            ent_type = {
                'PERSON': '👤',
                'ORGANIZATION': '🏢',
                'LOCATION': '📍',
                'EVENT': '📅',
                'PRODUCT': '📦',
                'TEMPORAL': '⏰'
            }.get(ent['type'], '📌')
            
            formatted_entities.append(f"{ent_type} {ent['text']}")
            
        return " | ".join(formatted_entities)
    
    @staticmethod
    def format_sentiment(sentiment: Dict) -> str:
        """Formate le sentiment pour l'affichage"""
        sentiment_icons = {
            'positive': '😊',
            'negative': '😔',
            'neutral': '😐'
        }
        
        label = sentiment.get('label', 'neutral')
        score = sentiment.get('score', 0)
        icon = sentiment_icons.get(label, '😐')
        
        return f"{icon} {label.capitalize()} ({score:.2f})"

class DataValidator:
    @staticmethod
    def validate_date_range(start_date: datetime, end_date: datetime) -> bool:
        """Valide une plage de dates"""
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            return False
        if start_date > end_date:
            return False
        if end_date > datetime.now():
            return False
        return True
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Nettoie un texte"""
        if not isinstance(text, str):
            return ""
        # Supprime les caractères spéciaux et les espaces multiples
        cleaned = ' '.join(text.split())
        return cleaned

class Config:
    # Paramètres de l'application
    APP_TITLE = "Analyse des Articles RTBF"
    LOGO_URL = "https://static-oaos.rtbf.be/icons/custom/brands/black/RTBFActus.svg"
    
    # Paramètres d'affichage
    MAX_ARTICLES_PER_PAGE = 20
    MAX_ENTITIES_DISPLAY = 3
    MAX_THEMES_DISPLAY = 3
    
    # Intervalles de temps prédéfinis
    TIME_RANGES = {
        "Aujourd'hui": 1,
        "Cette semaine": 7,
        "Ce mois": 30,
        "Cette année": 365
    }
    
    # Catégories de recherche
    SEARCH_CATEGORIES = {
        'general': 'Recherche générale',
        'person': 'Recherche de personnes',
        'location': 'Recherche par lieu',
        'temporal': 'Recherche temporelle',
        'event': 'Recherche d\'événements',
        'explanation': 'Recherche d\'explications'
    }