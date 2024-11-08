import requests
import csv
import time
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

class RTBFScraper:
    def __init__(self, base_url: str, output_file: str):
        self.base_url = base_url
        self.output_file = output_file
        self.root_url = "https://www.rtbf.be/article/"
        self.session = self._create_session()
        self._setup_logging()
        
    def _create_session(self) -> requests.Session:
        """Create a session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _setup_logging(self):
        """Setup logging configuration"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/scraping_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _fetch_page(self, page: int, limit: int = 100) -> Optional[Dict]:
        """Fetch a single page of articles"""
        url = f"{self.base_url}?_page={page}&_limit={limit}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching page {page}: {str(e)}")
            return None

    def _process_article(self, article: Dict) -> Dict:
        """Process a single article"""
        article_id = article.get('id')
        article_slug = article.get('slug')
        return {
            'id': article_id,
            'title': article.get('title'),
            'category': article.get('dossierLabel'),
            'date': article.get('publishedFrom'),
            'link': f"{self.root_url}{article_slug}-{article_id}",
            'summary': article.get('summary')
        }

    def scrape(self, pages: int, batch_size: int = 100) -> None:
        """Main scraping method"""
        self.logger.info(f"Starting scraping of {pages} pages")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Create/Open CSV file with headers
        fieldnames = ['id', 'title', 'summary', 'category', 'date', 'link']
        with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Scrape and write page by page
            for page in range(1, pages + 1):
                self.logger.info(f"Scraping page {page}/{pages}")
                
                data = self._fetch_page(page, batch_size)
                if not data:
                    continue
                    
                articles = data.get('data', {}).get('articles', [])
                processed_articles = [self._process_article(article) for article in articles]
                
                # Write articles to CSV
                writer.writerows(processed_articles)
                
                # Rate limiting
                time.sleep(1)  # Be nice to the server
                
        self.logger.info(f"Scraping completed. Data saved to {self.output_file}")

def main():
    # Configuration
    base_url = "https://bff-service.rtbf.be/oaos/v1.5/pages/en-continu"
    output_file = "data/raw/rtbf_articles.csv"
    pages_to_scrape = 100  # Adjust as needed to get more than 2000 articles
    
    # Initialize and run scraper
    scraper = RTBFScraper(base_url, output_file)
    scraper.scrape(pages_to_scrape)

if __name__ == "__main__":
    main()