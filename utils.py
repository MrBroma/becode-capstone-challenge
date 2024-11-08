import pandas as pd
from typing import Dict, List
from ast import literal_eval

def format_date(date_str: str, format: str = 'short') -> str:
    if pd.isna(date_str):
        return ""
    try:
        if isinstance(date_str, str):
            date = pd.to_datetime(date_str)
        else:
            date = date_str
        if format == 'short':
            return date.strftime('%d/%m/%Y')
        else:
            return date.strftime('%d/%m/%Y %H:%M')
    except:
        return ""

def extract_search_terms(query: str) -> str:
    """Extracts search terms from a natural language question"""
    question_prefixes = [
        "que peux tu me dire sur", "que sais tu sur",
        "parle moi de", "je voudrais des informations sur",
        "montre moi les articles sur", "recherche des articles sur",
        "que s'est il passé avec", "quoi de neuf sur",
        "qu'est ce qui concerne", "peux tu me parler de",
        "articles sur", "infos sur", "actualités sur"
    ]
    
    query = query.lower().strip()
    
    for prefix in question_prefixes:
        if query.startswith(prefix):
            return query[len(prefix):].strip()
    
    return query

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """Loads and preprocesses the data"""
    df = pd.read_csv(file_path)
    
    # Convertir la colonne entities
    df['entities'] = df['entities'].apply(lambda x: [] if pd.isna(x) else literal_eval(x))
    
    # Traiter les dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        df = df.dropna(subset=['date'])
        df = df.sort_values('date', ascending=False)
    
    return df

def calculate_relevance(df: pd.DataFrame, search_term: str) -> pd.DataFrame:
    """Calculates the relevance score of the articles"""
    df = df.copy()
    df['relevance'] = (
        df['title'].str.contains(search_term, case=False).astype(int) * 2 +
        df['summary'].str.contains(search_term, case=False).astype(int)
    ) / 3 * 100
    return df.sort_values('relevance', ascending=False)

def create_search_mask(df: pd.DataFrame, search_term: str = "", category: str = "Toutes", days_filter: int = None) -> pd.Series:
    """Crée un masque de filtrage pour la recherche"""
    mask = pd.Series(True, index=df.index)
    
    if search_term:
        mask &= (
            df['title'].str.contains(search_term, case=False, na=False) |
            df['summary'].str.contains(search_term, case=False, na=False)
        )
    
    if category != 'Toutes':
        mask &= (df['category'] == category)
    
    if days_filter and 'date' in df.columns:
        cutoff_date = pd.Timestamp.now().tz_localize(None) - pd.Timedelta(days=days_filter)
        mask &= (df['date'] > cutoff_date)
    
    return mask