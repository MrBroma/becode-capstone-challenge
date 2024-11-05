import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def get_page_content(url, page=1):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Ajouter le paramètre de page à l'URL
        full_url = f"{url}?page={page}" if page > 1 else url
        response = requests.get(full_url, headers=headers)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"Erreur lors de la récupération de la page {page}: {e}")
        return None

def extract_article_data(article):
    title_tag = article.find('a', class_='stretched-link')
    title = title_tag.text.strip() if title_tag else 'N/A'
    
    footer_date_tag = article.find('div', class_='card-meta')
    time_tag = footer_date_tag.find('time') if footer_date_tag else None
    footer_date = time_tag.get('datetime') if time_tag else 'N/A'
    
    category_tag = article.find('div', class_='inline-flex')
    category = category_tag.text.strip() if category_tag else 'N/A'
    
    link_tag = article.find('a')
    link = 'https://www.rtbf.be' + link_tag.get('href') if link_tag else 'N/A'
    
    return {
        'Title': title,
        'Date': footer_date,
        'Category': category,
        'Link': link
    }

# URL de base
raw_url = 'https://www.rtbf.be/en-continu'
data = []
page = 1
max_pages = 10  # Limite de pages à scraper

while page <= max_pages:
    print(f"Récupération de la page {page}...")
    soup = get_page_content(raw_url, page)
    
    if not soup:
        break
        
    articles = soup.find_all('article')
    
    if not articles:
        print("Plus d'articles trouvés")
        break
        
    for article in articles:
        article_data = extract_article_data(article)
        data.append(article_data)
    
    page += 1
    time.sleep(2)  # Pause de 2 secondes entre chaque requête

# Créer le DataFrame
df = pd.DataFrame(data)

# Supprimer les doublons éventuels
df = df.drop_duplicates()

# Afficher le nombre total d'articles récupérés
print(f"\nNombre total d'articles récupérés : {len(df)}")

# Afficher le DataFrame
print(df)

# Enregistrer dans un fichier CSV
df.to_csv('articles.csv', index=False)