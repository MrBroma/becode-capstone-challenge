import requests
import csv
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def scrape_page(page: int, base_url: str) -> List[dict]:
    url = f"{base_url}?_page={page}&_limit=100"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        articles = data.get('data', {}).get('articles', [])
        return [
            {
                'id': article.get('id'),
                'title': article.get('title'),
                'category': article.get('dossierLabel'),
                'date': article.get('publishedFrom'),
                'link': f"https://www.rtbf.be/article/{article.get('slug')}-{article.get('id')}",
                'summary': article.get('summary')
            }
            for article in articles
        ]
    else:
        print(f"Failed to fetch page {page}: {response.status_code}")
        return []

def scrape_rtbf_data(base_url, pages, output_file):
    all_articles = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(scrape_page, page, base_url) for page in range(1, pages + 1)]
        
        for future in tqdm(as_completed(futures), total=pages, unit='page'):
            articles = future.result()
            all_articles.extend(articles)

    # Write data to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'title', 'summary', 'category', 'date', 'link']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for article in all_articles:
            writer.writerow(article)

    print(f"Scraping completed. Data saved to {output_file}")
    return all_articles

def get_top_articles(n: int) -> List[dict]:
    """Récupère les n articles les plus récents"""
    base_url = "https://bff-service.rtbf.be/oaos/v1.5/pages/en-continu"
    pages_to_scrape = n // 100 + 1
    articles = scrape_rtbf_data(base_url, pages_to_scrape, "data/rtbf_articles.csv")
    return articles[:n]

if __name__ == "__main__":
    # Modify the variables as needed
    base_url = "https://bff-service.rtbf.be/oaos/v1.5/pages/en-continu"
    pages_to_scrape = 100  # Change this to scrape more or fewer pages
    output_csv = "data/rtbf_articles.csv"
    scrape_rtbf_data(base_url, pages_to_scrape, output_csv)