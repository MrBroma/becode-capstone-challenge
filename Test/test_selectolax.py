import json
from selectolax.parser import HTMLParser
import requests

def scrape(url):
    try:
        # Envoyer une requête GET à l'URL
        response = requests.get(url)
        response.raise_for_status()  # Vérifie que la requête s'est bien passée
        
        # Parser le contenu HTML avec selectolax
        html = HTMLParser(response.text)
        
        # print soup html
        print(html.body.text())

        # Chercher toutes les balises 'article' ou autre contenu pertinent
        articles = html.css('article')
        
        # Liste pour stocker les données extraites
        data = []
        
        # Boucler à travers les articles trouvés
        for article in articles:
            # Extraire des informations comme le titre, la date, le lien, etc.
            title_tag = article.css_first('a.stretched-link')
            title = title_tag.text(strip=True) if title_tag else 'N/A'
            
            footer_date_tag = article.css_first('div.card-meta time')
            footer_date = footer_date_tag.attrs.get('datetime') if footer_date_tag else 'N/A'
            
            category_tag = article.css_first('div.inline-flex')
            category = category_tag.text(strip=True) if category_tag else 'N/A'
            
            link_tag = article.css_first('a')
            link = f"https://www.rtbf.be{link_tag.attrs.get('href')}" if link_tag else 'N/A'
            
            # Ajouter les informations extraites à la liste de données
            data.append({
                'Title': title,
                'Date': footer_date,
                'Category': category,
                'Link': link
            })
        
        return data
    except Exception as e:
        print(f"An error occurred while scraping {url}: {e}")
        return None

# Exemple d'utilisation
raw_url = 'https://www.rtbf.be/en-continu'
scraped_data = scrape(raw_url)

if scraped_data:
    print(json.dumps(scraped_data, indent=4))

   