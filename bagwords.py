import pathlib
import re
from collections import Counter


def lire_mots_vides(fichier_path):
    """Lit un fichier contenant les mots vides (un par ligne)"""
    with open(fichier_path, encoding="utf-8") as f:
        return set(ligne.strip() for ligne in f if ligne.strip())


def nettoyage(texte):
    """Nettoie le texte (ponctuation, chiffres, etc.)"""
    texte = re.sub(r"[&%!?\|_\"(){}\[\],.;/\\:§»«”“‘…–—−]+", " ", texte)
    texte = re.sub(r"\d+", "", texte)
    texte = texte.replace("\\", " ")
    texte = texte.replace("’", "'")
    texte = texte.replace("'", "' ")
    return texte


def bag_of_words(text, mots_vides):
    """Construit le bag of words à partir d'un texte brut"""
    text = nettoyage(text)
    mots = text.lower().split()
    return Counter(m for m in mots if m and m not in mots_vides)


def traiter_corpus(corpus_dir, mots_vides):
    """Traite le corpus organisé en sous-dossiers (un par classe)"""
    dossier = pathlib.Path(corpus_dir)
    resultats = []

    for classe in sorted(d for d in dossier.iterdir() if d.is_dir()):
        for fichier in classe.glob("*.txt"):
            texte = fichier.read_text(encoding="utf-8")
            bow = bag_of_words(texte, mots_vides)
            resultats.append((classe.name, fichier.name, dict(bow)))

    return resultats


def sauvegarder_resultats(resultats, output_path):
    """Sauvegarde les résultats dans un fichier texte"""
    with open(output_path, "w", encoding="utf-8") as out:
        for classe, nom_fichier, bow in resultats:
            out.write(f"Classe: {classe}, Fichier: {nom_fichier}\n")
            for mot, freq in sorted(bow.items(), key=lambda x: (-x[1], x[0])):
                out.write(f"  {mot}: {freq}\n")
            out.write("\n")


if __name__ == "__main__":
    corpus_path = "corpus_livres3"  
    fichier_mots_vides = "/home/ines/GoodRScrape/mots_vides.txt"  
    output_file = "resultats_bag_of_words.txt"

    mots_vides = lire_mots_vides(fichier_mots_vides)
    resultats = traiter_corpus(corpus_path, mots_vides)
    sauvegarder_resultats(resultats, output_file)

    print(f"✅ Résultats enregistrés dans : {output_file}")

