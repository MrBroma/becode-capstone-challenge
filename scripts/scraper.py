import requests
import csv

def scrape_rtbf_data(base_url, pages, output_file):
    all_articles = []
    root_url = "https://www.rtbf.be/article/"
    
    # Loop through the specified number of pages
    for page in range(1, pages + 1):
        print(f"Scraping page {page}/{pages}...")
        url = f"{base_url}?_page={page}&_limit=100"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('data', {}).get('articles', [])
            for article in articles:
                article_id = article.get('id')
                article_slug = article.get('slug')
                article_url = f"{root_url}{article_slug}-{article_id}"
                
                article_data = {
                    'id': article_id,
                    'title': article.get('title'),
                    'category': article.get('dossierLabel'),
                    'date': article.get('publishedFrom'),
                    'link': article_url
                }
                all_articles.append(article_data)
        else:
            print(f"Failed to fetch page {page}: {response.status_code}")

    # Write data to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'title', 'category', 'date', 'link']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for article in all_articles:
            writer.writerow(article)

    print(f"Scraping completed. Data saved to {output_file}")

# Modify the variables as needed
base_url = "https://bff-service.rtbf.be/oaos/v1.5/pages/en-continu"
pages_to_scrape = 50  # Change this to scrape more or fewer pages
output_csv = "rtbf_articles.csv"

scrape_rtbf_data(base_url, pages_to_scrape, output_csv)


