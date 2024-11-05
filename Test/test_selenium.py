from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

# Configurer Selenium pour utiliser Chrome
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Exécuter en mode headless
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

raw_url = 'https://www.rtbf.be/en-continu'
driver.get(raw_url)

# Fonction pour accepter ou fermer la fenêtre des cookies
def handle_cookie_banner():
    try:
        # Attendre que la fenêtre des cookies soit cliquable et cliquer sur le bouton 'Accepter'
        cookie_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accepter')]"))  # Modifiez ce sélecteur si nécessaire
        )
        cookie_button.click()
        print("Fenêtre des cookies fermée.")
    except Exception as e:
        print(f"Erreur lors de la gestion de la fenêtre des cookies: {e}")

# Gérer la fenêtre des cookies
handle_cookie_banner()

# Fonction pour récupérer les articles
def get_articles():
    articles = driver.find_elements(By.TAG_NAME, 'article')
    data = []
    for article in articles:
        try:
            title_tag = article.find_element(By.CLASS_NAME, 'stretched-link')
            title = title_tag.text.strip() if title_tag else 'N/A'

            footer_date_tag = article.find_element(By.CLASS_NAME, 'card-meta')
            time_tag = footer_date_tag.find_element(By.TAG_NAME, 'time')
            footer_date = time_tag.get_attribute('datetime') if time_tag else 'N/A'

            category_tag = article.find_element(By.CLASS_NAME, 'inline-flex')
            category = category_tag.text.strip() if category_tag else 'N/A'

            link_tag = article.find_element(By.TAG_NAME, 'a')
            link = 'https://www.rtbf.be' + link_tag.get_attribute('href') if link_tag else 'N/A'

            data.append({
                'Title': title,
                'Date': footer_date,
                'Category': category,
                'Link': link
            })
        except Exception as e:
            print(f"Erreur lors de la récupération d'un article: {e}")
    return data

# Récupérer les articles initiaux
data = get_articles()
print("Récupération initiale d'articles terminée.")
print(f"Nombre d'articles récupérés : {len(data)}")

# Automatiser le chargement de plus d'articles
scroll_height = driver.execute_script("return document.body.scrollHeight;")
while len(data) < 500:
    try:
        # Faire défiler la page pour charger plus d'articles
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Attendre le chargement des articles
        new_scroll_height = driver.execute_script("return document.body.scrollHeight;")
        if new_scroll_height == scroll_height:
            break  # Plus d'articles à charger
        scroll_height = new_scroll_height

        new_articles = get_articles()
        data.extend(new_articles)
        print(f"Nombre total d'articles récupérés : {len(data)}")
    except Exception as e:
        print(f"Erreur lors du chargement de plus d'articles: {e}")
        break

# Limiter à 500 articles
if len(data) > 500:
    data = data[:500]

# Créer le DataFrame
df = pd.DataFrame(data)

# Afficher le DataFrame
print(df)

# Enregistrer dans un fichier CSV
df.to_csv('articles.csv', index=False)

# Fermer le driver
driver.quit()