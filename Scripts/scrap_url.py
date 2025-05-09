import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Ce script à permis de recupérer les urls des livres

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
}


def get_book_urls_from_genre(genre_url, max_pages=5):
    """Récupère les URLs des livres depuis les pages de genre"""
    book_urls = []
    for page in range(1, max_pages + 1):
        url = f"{genre_url}/{page}"
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extraire les liens des livres (à adapter au HTML réel)
        for book in soup.select("a.bookTitle"):
             book_urls.append("https://www.goodreads.com" + book["href"])

        
        time.sleep(2)  # Délai pour éviter le blocage
    
    return book_urls


genre_urls = {
    "Fantasy": "https://www.goodreads.com/shelf/show/fantasy",
    "Poésie": "https://www.goodreads.com/shelf/show/poetry",
    "Comédie": "https://www.goodreads.com/shelf/show/comics"
}

all_urls = {}
for genre, url in genre_urls.items():
    all_urls[genre] = get_book_urls_from_genre(url, max_pages=10)  # 10 pages = ~100 livres
    print(f"{len(all_urls[genre])} URLs récupérées pour {genre}")

# Sauvegarde des URLs
data = []
for genre, urls in all_urls.items():
    for url in urls:
        data.append({"Genre": genre, "URL": url})

pd.DataFrame(data).to_csv("book_urls_by_genre.csv", index=False)