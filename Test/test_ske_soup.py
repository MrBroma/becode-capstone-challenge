import requests
import xml.etree.ElementTree as ET
import csv

# URL du sitemap
sitemap_url = 'https://www.rtbf.be/site-map/articles5000.xml'

# Faire une requête GET pour récupérer le contenu du sitemap
response = requests.get(sitemap_url)

# Vérifier que la requête a réussi
if response.status_code == 200:
    # Parser le contenu XML
    root = ET.fromstring(response.content)

    # Créer une liste pour stocker les articles
    articles = []

    # Espaces de noms
    namespaces = {
        'news': 'http://www.google.com/schemas/sitemap-news/0.9',
        'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9',
        'image': 'http://www.google.com/schemas/sitemap-image/1.1',
    }

    # Parcourir chaque élément <url> dans le XML
    for url in root.findall('{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
        # Extraire les informations
        loc = url.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc').text
        title = url.find('.//{http://www.google.com/schemas/sitemap-news/0.9}title').text
        publication_date = url.find('.//{http://www.google.com/schemas/sitemap-news/0.9}publication_date').text
        
        # Ajouter les informations extraites à la liste des articles
        articles.append({
            'title': title,
            'publication_date': publication_date,
            'link': loc,           
        })

    # Enregistrer les articles dans un fichier CSV
    with open('articles_rtbf.csv', mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['title', 'publication_date', 'link']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Écrire l'en-tête
        writer.writeheader()

        # Écrire les lignes
        for article in articles:
            writer.writerow(article)

    print("Les articles ont été enregistrés dans 'articles_rtbf.csv'.")

else:
    print(f"Erreur lors de la récupération du sitemap: {response.status_code}")
