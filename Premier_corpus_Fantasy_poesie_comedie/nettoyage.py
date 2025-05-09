import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import os
import re
import spacy

# Fonction pour charger les mots vides à partir d'un fichier
def read_lines_set(filename, encoding="utf-8"):
    with open(filename, encoding=encoding) as in_stream:
        return set(line.strip() for line in in_stream)

def nettoyage(texte):
    texte = texte.lower()
    texte = re.sub(r"http\S+", " ", texte)  # Supprimer URLs
    texte = re.sub(r"#\S+", " ", texte)     # Supprimer hashtags
    texte = re.sub(r"@\S+", " ", texte)     # Supprimer mentions
    texte = re.sub(r"[^\w\s]", " ", texte)  # Supprimer ponctuation
    texte = re.sub(r"\d+", " ", texte)      # Supprimer chiffres
    texte = re.sub(r"_", " ", texte)        # Supprimer underscores
    texte = re.sub(r"\s+", " ", texte)
    texte = re.sub(r"^\s+|\s+$", "", texte)

    # Supprimer les répétitions absurdes
    texte = re.sub(r"\b([a-z])\1{4,}\b", " ", texte)
    texte = re.sub(r"\b([a-z]+)\1{2,}\b", " ", texte)

    # Supprimer les mots en arabe et en cyrillique
    texte = re.sub(r"[\u0600-\u06FF]+", " ", texte)
    texte = re.sub(r"[\u0400-\u04FF]+", " ", texte)

    # Garder uniquement les mots avec uniquement lettres latines (a-z) ET longueur ≤ 10
    mots_filtres = [
        mot for mot in texte.split()
        if re.fullmatch(r"[a-z]+", mot) and len(mot) <= 10
    ]
    texte = " ".join(mots_filtres)

    return texte.strip()

# Charger le modèle spaCy en anglais
nlp = spacy.load("en_core_web_sm")

def lemmatisation_verbale_base_en(texte):
    doc = nlp(texte)
    return [
        token.lemma_.lower()
        for token in doc
        if token.pos_ == "VERB" and token.lemma_.isalpha()
    ]

# Variable pour les types de données valides
DATATYPES = ["tokens", "caractères"]
def bag_of_words(text, mots_vides, datatype="tokens"):
    text = nettoyage(text)
    if datatype == "caractères":
        # Ici on renvoie les caractères sous forme de chaîne
        return "".join(w for w in text if w and not w.isspace() and w not in mots_vides)
    elif datatype == "tokens":
        mots_lemmatises = lemmatisation_verbale_base_en(text)
        # Ici on renvoie les mots sous forme de chaîne
        return " ".join(w for w in mots_lemmatises if w not in mots_vides)
    elif datatype not in DATATYPES:
        raise ValueError(f"datatype incorrecte: {datatype}, attendu: {DATATYPES}")
    else:
        raise NotImplementedError(f"datatype non gérée: {datatype}")


# Fonction principale pour exécuter le processus d'analyse
def main():
    mots_vides = read_lines_set("*/GoodRScrape/Premier_corpus_Fantasy_poesie_comedie/mots_vides.txt")  # Remplacez par le chemin vers votre fichier de mots vides

    # Chargement du dataset
    df = pd.read_csv("metadata_eng.csv")  
    df["resume_clean"] = df["resume"].apply(lambda x: bag_of_words(x, mots_vides))

    # Vectorisation TF-IDF
    print("Vectorisation TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=None,
        stop_words="english",  # Vous pouvez garder cela si vous voulez également les mots vides anglais de base
        ngram_range=(1, 3),
        min_df=5,
        max_df=0.85,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(df["resume_clean"])
    print(f"Nombre total de mots utilisés dans l’analyse TF-IDF : {len(vectorizer.get_feature_names_out())}")

    y = df["genre"]

    # Split en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    models = {
        "SVM": GridSearchCV(SVC(kernel="linear"), {"C": [0.1, 1, 10]}, cv=5),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Random Forest": GridSearchCV(
            RandomForestClassifier(), {"n_estimators": [50, 100], "max_depth": [None, 10]}, cv=3
        )
    }

    os.makedirs("results", exist_ok=True)
    results = {}

    for name, model in models.items():
        print(f"Évaluation du modèle {name}")
        model.fit(X_train, y_train)

        if hasattr(model, "best_params_"):
            print(f"Meilleurs paramètres : {model.best_params_}")

        y_pred = model.predict(X_test)
        print("Rapport de classification :")
        print(classification_report(y_test, y_pred, digits=3))

        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = report

        # Matrice de confusion
        plt.figure(figsize=(10, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f"Matrice de confusion - {name}\nAccuracy: {report['accuracy']:.3f}")
        plt.tight_layout()
        plt.savefig(f"results/confusion_{name.lower()}.png", dpi=300)
        plt.close()

        if name == "Random Forest":
            print("Top 20 des mots les plus discriminants :")
            feat_imp = pd.DataFrame({
                "feature": vectorizer.get_feature_names_out(),
                "importance": model.best_estimator_.feature_importances_
            }).sort_values("importance", ascending=False)
            print(feat_imp.head(20))

            feat_imp.head(30).to_csv("results/top_features.csv", index=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(data=feat_imp.head(20), x="importance", y="feature")
            plt.title("Top 20 des mots discriminants")
            plt.tight_layout()
            plt.savefig("results/top_features.png", dpi=300)
            plt.close()

    pd.DataFrame.from_dict(results, orient="index").to_csv("results/classification_report.csv")
    print("Analyse terminée. Résultats sauvegardés dans /results")

# Lancer le processus principal
if __name__ == "__main__":
    main()
