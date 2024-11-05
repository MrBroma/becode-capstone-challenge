import requests
from bs4 import BeautifulSoup

# URL of the page to scrape
url = 'https://www.rtbf.be/en-continu'

# Sending a request to fetch the content of the page
response = requests.get(url)
html_content = response.content


# Parsing the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Finding all articles based on the title link
articles = soup.find_all('a', class_='stretched-link')

print(articles)

# save it in a text file
with open('articles.txt', 'w') as file:
    for article in articles:
        file.write(article.text + '\n')


