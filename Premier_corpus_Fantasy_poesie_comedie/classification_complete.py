# classification_complete.py

import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Partie 1 : Nettoyage
# --------------------------

def charger_mots_interdits(dossier="mots_interdits"):
    """
    Coucou ! Ici on charge deux listes : les mots parasites (genre 'www', 'http') 
    et les mots trop évidents comme 'book' ou 'novel' qui n’aident pas à classifier.
    Si les fichiers sont introuvables, on a une version de secours en dur.
    """
    mots_parasites = set()
    mots_evidents = set()

    try:
        with open(Path(dossier) / "mots_parasites.txt", 'r', encoding='utf-8') as f:
            mots_parasites = {ligne.strip() for ligne in f if ligne.strip()}

        with open(Path(dossier) / "mots_evidents.txt", 'r', encoding='utf-8') as f:
            mots_evidents = {ligne.strip() for ligne in f if ligne.strip()}

    except FileNotFoundError:
        # Plan B si les fichiers sont absents
        mots_parasites = {"www", "http", "html", "img"}
        mots_evidents = {"book", "novel", "story", "author"}

    return mots_parasites, mots_evidents

def nettoyage_avance(texte):
    """
    On nettoie le texte à fond ici : tout passe en minuscule, on enlève les chiffres,
    les mots trop courts, les caractères chelous, etc.
    Ensuite on vire les mots parasites et trop évidents.
    """
    mots_parasites, mots_evidents = charger_mots_interdits()

    if not isinstance(texte, str):
        return ""

    texte = texte.lower()
    texte = re.sub(r"[^\w\s']", " ", texte)  # on garde que les lettres/chiffres/apostrophes
    texte = re.sub(r"\d+", " ", texte)       # on enlève tous les chiffres

    # On garde que les mots utiles (genre pas trop courts, pas bizarres, pas dans les listes)
    mots = [
        mot for mot in texte.split()
        if (len(mot) > 2
            and re.fullmatch(r"[a-z']+", mot)
            and mot not in mots_parasites
            and mot not in mots_evidents
            and not any(c.isdigit() for c in mot))
    ]

    return " ".join(mots)

# --------------------------
# Partie 2 : Classification
# --------------------------
def main():
    # 1. On commence par charger les données
    try:
        df = pd.read_csv("metadata_eng.csv")
        print(f"✅ Corpus chargé : {len(df)} livres")
        print("Répartition des genres :")
        print(df["genre"].value_counts())
    except Exception as e:
        print(f"❌ Erreur : {str(e)}")
        return

    # 2. Hop, on nettoie tous les résumés !
    print(" Nettoyage des textes...")
    df["resume_clean"] = df["resume"].apply(nettoyage_avance)

    # 3. Maintenant on transforme les textes en vecteurs numériques avec TF-IDF
    print(" Vectorisation TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=1500,          # max 1500 mots conservés
        stop_words="english",       # suppression des mots vides
        ngram_range=(1, 3),         # on garde les mots seuls, mais aussi les groupes de 2-3
        min_df=5,                   # un mot doit apparaître dans au moins 5 textes
        max_df=0.85,                # mais pas dans plus de 85% des textes (trop courant sinon)
        sublinear_tf=True           # log(1 + tf), pour écraser les valeurs trop grandes
    )

    X = vectorizer.fit_transform(df["resume_clean"])  # Les vecteurs des textes
    y = df["genre"]  # Les étiquettes (les genres)

    # 4. Split en train / test pour évaluer nos modèles comme des pros
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # 5. On définit trois modèles qu'on veut tester
    models = {
        "SVM": GridSearchCV(
            SVC(kernel="linear"),
            {"C": [0.1, 1, 10]},  # test de plusieurs valeurs de C
            cv=5
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Random Forest": GridSearchCV(
            RandomForestClassifier(),
            {"n_estimators": [50, 100], "max_depth": [None, 10]},
            cv=3
        )
    }

    # 6. On entraîne chaque modèle et on affiche les résultats
    results = {}
    for name, model in models.items():
        print(f" Évaluation du modèle {name}")
        model.fit(X_train, y_train)

        # Si y a eu un GridSearch, on affiche les meilleurs paramètres
        if hasattr(model, "best_params_"):
            print(f"Meilleurs paramètres : {model.best_params_}")

        y_pred = model.predict(X_test)

        # On affiche les scores (précision, rappel, f-mesure)
        print(" Rapport de classification :")
        print(classification_report(y_test, y_pred, digits=3))

        # On sauvegarde les scores dans un dico
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = report

        # On affiche une matrice de confusion trop stylée
        plt.figure(figsize=(10, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=model.classes_,
                    yticklabels=model.classes_)
        plt.title(f"Matrice de confusion - {name}\nAccuracy: {report['accuracy']:.3f}")
        plt.tight_layout()
        plt.savefig(f"results/confusion_{name.lower()}.png", dpi=300)
        plt.close()

        # 7. Si on est avec la Random Forest, on va analyser les mots les plus importants
        if name == "Random Forest":
            print(" Top 20 des mots les plus discriminants :")
            feat_imp = pd.DataFrame({
                "feature": vectorizer.get_feature_names_out(),
                "importance": model.best_estimator_.feature_importances_
            }).sort_values("importance", ascending=False)
            print(feat_imp.head(20))

            # Bonus : on vérifie que certains mots qu'on soupçonne sont bien utiles
            print(" Validation manuelle des mots clés :")
            mots_a_verifier = ["graphic", "volume", "art", "character"]
            for mot in mots_a_verifier:
                print(f" Distribution du mot '{mot}' par genre :")
                print(df[df['resume_clean'].str.contains(mot, na=False)]['genre'].value_counts())
                print("Exemples :")
                print(df[df['resume_clean'].str.contains(mot, na=False)].sample(2)[['genre', 'resume_clean']])
            
            feat_imp.head(30).to_csv("results/top_features.csv", index=False)

            # On fait un joli graphe des mots top
            plt.figure(figsize=(12, 8))
            sns.barplot(data=feat_imp.head(20), x="importance", y="feature")
            plt.title("Top 20 des mots discriminants")
            plt.tight_layout()
            plt.savefig("results/top_features.png", dpi=300)
            plt.close()

    # 8. On sauvegarde tous les rapports pour plus tard
    pd.DataFrame.from_dict(results, orient="index").to_csv("results/classification_report.csv")
    print(" Analyse terminée. Résultats sauvegardés dans /results")

# Script lancé directement ?
if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)  # si le dossier results n'existe pas, on le crée
    main()

