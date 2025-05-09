import os

def compter_fichiers_par_sous_dossier(dossier):
    try:
        for racine, dossiers, fichiers in os.walk(dossier):
            nombre_fichiers = len([f for f in fichiers if os.path.isfile(os.path.join(racine, f))])
            print(f"{racine} : {nombre_fichiers} fichier(s)")
    except FileNotFoundError:
        print(f"Le dossier '{dossier}' n'existe pas.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

# Exemple d'utilisation :
dossier_cible = "corpus_livres"  # Remplace par le chemin réel
compter_fichiers_par_sous_dossier(dossier_cible)
