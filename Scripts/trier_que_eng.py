import os
import pandas as pd

# Après le scrapping je me suis rendue compte que plusieurs livres n'était pas en anglais 
# Et nous avons choisi de garder uniquement ceux en anglais 

# Liste de mots très fréquents en anglais (stopwords)
# → genre pour détecter si un texte est en anglais (j'ai fait ça parce que c'est le plus simple)
common_english_words = {
    "the", "and", "is", "to", "in", "that", "with", "he", "she", "his", "her",
    "of", "a", "as", "for", "this", "you", "but", "they", "their", "who"
}

def is_english(text):
    """Renvoie True si le texte contient suffisamment de mots anglais fréquents."""
    if not isinstance(text, str):
        return False  # Si le texte est vide ou pas une string, on zappe
    words = text.lower().split()
    if len(words) == 0:
        return False  # Si y'a rien à tester, ben on considère que c'est pas de l'anglais
    # On compte le nombre de mots dans la liste des mots fréquents
    common_count = sum(1 for word in words if word in common_english_words)
    return common_count / len(words) > 0.05  # Si y'a plus de 5% de mots fréquents, on garde

def main():
    # Charger le fichier CSV avec tous les livres scrapés
    df = pd.read_csv("/Users/oumayachelbi/Desktop/M1S2/fouille/corpus_livres3/metadata_all.csv")

    # Filtrer les lignes pour garder que les résumés en anglais
    df_en = df[df["resume"].apply(is_english)]

    # Juste pour voir combien on en a par genre après le tri
    print("\nNombre de livres en anglais par genre :")
    print(df_en["genre"].value_counts())

    # On enregistre ce nouveau sous-ensemble dans un nouveau fichier CSV
    df_en.to_csv("metadata_en.csv", index=False)

    # Ensuite on recrée tous les fichiers .txt (résumés) mais seulement pour les livres en anglais
    for _, row in df_en.iterrows():
        titre = row["titre"]
        resume = row["resume"]
        genre = row["genre"]

        # Nettoyage des noms pour pas qu’il y ait des / dans les noms de dossiers ou fichiers
        genre_clean = genre.strip().replace("/", "-")
        titre_clean = titre.strip().replace("/", "-")

        # Création du chemin du dossier
        output_path = os.path.join("corpus_livres_english3", genre_clean)
        os.makedirs(output_path, exist_ok=True)  # Si le dossier existe pas, on le crée

        file_path = os.path.join(output_path, f"{titre_clean}.txt")

        # Écriture du résumé dans un fichier texte
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(resume)

if __name__ == "__main__":
    main()

