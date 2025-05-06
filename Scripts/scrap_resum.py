import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import re

# Ce script à permis de scrapper et garder les infos qu'on voulait à savoir le titre
# des livre, leurs résumer et les stockés dans le dossier correspondant à leur genre

# Configuration
HEADERS = {
    # Juste un user-agent pour éviter de se faire bloquer par Goodreads, j'ai fait ça après avoir 
    # lu le fichier robot.txt comme on l'a vu en Outils Traitment Corpus
    "User-Agent": "Mozilla/5.0 (compatible; MasterTAL-Project/1.0; +https://univ-paris3.fr)"
}
DELAY = 3  # Pause de 3 secondes entre chaque requête pour pas abuser
OUTPUT_DIR = "corpus_livres3"  # Dossier où on stocke tout

def create_directory_structure(genres):
    # Crée les dossiers pour chaque genre (ex : corpus_livres3/Romance)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for genre in genres:
        os.makedirs(os.path.join(OUTPUT_DIR, genre), exist_ok=True)

def sanitize_filename(title):
    # Nettoie les titres pour en faire des noms de fichiers valides (pas de caractères chelous)
    return re.sub(r'[\\/*?:"<>|]', "", title)[:100]

def scrape_book_page(url, genre):
    """Scrape le titre et le résumé depuis une page Goodreads"""
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()  # Si jamais la page répond pas, ça lève une erreur
        soup = BeautifulSoup(response.text, 'html.parser')

        # On choppe le titre (c'est souvent dans une balise h1 avec une certaine classe)
        titre_element = soup.find("h1", {"class": "Text__title1"})
        if not titre_element:
            print(f"Pas de titre trouvé pour {url}")
            return None
        titre = titre_element.get_text(strip=True)

        # Ensuite on chope le résumé. J'ai trouver en inspectant la page (cmd+option+I) que c’est dans des balises <span> avec la classe "Formatted"
        resume_elements = soup.find_all("span", class_="Formatted")
        if resume_elements:
            # On prend celui qui a le plus de texte, ça évite de prendre un mini extrait
            resume_element = max(resume_elements, key=lambda x: len(x.get_text(strip=True)))
            resume = resume_element.get_text(strip=True)
        else:
            resume = "Pas de résumé disponible."

        # On enregistre dans un fichier texte dans le bon dossier
        filename = f"{sanitize_filename(titre)}.txt"
        filepath = os.path.join(OUTPUT_DIR, genre, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(resume)

        # On retourne un petit dict avec les infos qu’on garde aussi dans un CSV à la fin
        return {
            "titre": titre,
            "resume": resume,
            "genre": genre,
            "url": url,
            "fichier": filename
        }

    except Exception as e:
        # Si y’a une erreur (genre page introuvable), on affiche juste un message
        print(f"Erreur sur {url} : {str(e)}")
        return None

def main():
    # On lit le fichier CSV avec les URLs à scraper (faut que ce fichier existe et ait les bonnes colonnes)
    df_urls = pd.read_csv("book_urls_by_genre3.csv")

    if 'Genre' not in df_urls.columns or 'URL' not in df_urls.columns:
        print("Le fichier CSV doit contenir les colonnes 'Genre' et 'URL'.")
        return

    # On prépare la liste des couples (genre, URL)
    books_to_scrape = list(df_urls.itertuples(index=False, name=None))

    # On récupère tous les genres uniques pour créer les dossiers
    genres = df_urls['Genre'].unique().tolist()
    create_directory_structure(genres)

    all_books = []
    for genre, url in books_to_scrape:
        print(f"Traitement : {url}")  # Affiche l’URL en cours pour suivre
        book_data = scrape_book_page(url, genre)
        if book_data:
            all_books.append(book_data)
        time.sleep(DELAY)  # Petite pause pour pas se faire bloquer par Goodreads

    # Une fois que tout est scrapé, on crée un CSV récapitulatif
    df = pd.DataFrame(all_books)
    df.to_csv(os.path.join(OUTPUT_DIR, "metadata_all.csv"), index=False, encoding='utf-8')
    print(f"Terminé ! {len(df)} livres scrapés et enregistrés.")

if __name__ == "__main__":
    main()
