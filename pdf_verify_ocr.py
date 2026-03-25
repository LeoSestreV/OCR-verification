#!/usr/bin/env python3
"""
pdf_verify_ocr.py — Script de vérification OCR pour volumes de biographies nationales en PDF.

Ce script :
1. Extrait le texte de chaque PDF via PyMuPDF (extraction de base).
2. Envoie le PDF à l'API Mistral OCR pour une transcription haute précision.
3. Compare les deux résultats terme à terme.
4. Génère un fichier d'erreurs nommé par le nombre de termes non reconnus.

Dossiers utilisés :
    /source_pdf/   — PDF originaux
    /output_txt/   — Textes issus de l'extraction de base (PyMuPDF)
    /ocr_mistral/  — Résultats complets de Mistral OCR
    /erreurs/      — Fichiers nommés par le nombre d'erreurs

Utilisation :
    export MISTRAL_API_KEY="votre_clé"
    python pdf_verify_ocr.py [--source SOURCE] [--seuil 0.75]

Dépendances :
    pip install -r requirements.txt
"""

import os
import sys
import argparse
import base64
import re
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

import fitz  # PyMuPDF
from mistralai import Mistral
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paramètres par défaut
# ---------------------------------------------------------------------------
SEUIL_SIMILARITE = 0.75  # En dessous de ce seuil, le terme est considéré non reconnu
DOSSIER_SOURCE = "source_pdf"
DOSSIER_OUTPUT = "output_txt"
DOSSIER_MISTRAL = "ocr_mistral"
DOSSIER_ERREURS = "erreurs"


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def creer_dossiers(racine: Path) -> None:
    """Crée les dossiers de sortie s'ils n'existent pas."""
    for nom in [DOSSIER_SOURCE, DOSSIER_OUTPUT, DOSSIER_MISTRAL, DOSSIER_ERREURS]:
        (racine / nom).mkdir(parents=True, exist_ok=True)


def normaliser_texte(texte: str) -> str:
    """Normalise le texte : minuscules, suppression des accents parasites,
    remplacement des ligatures et nettoyage des caractères spéciaux."""
    # Minuscules
    texte = texte.lower()
    # Normalisation Unicode (décomposition canonique)
    texte = unicodedata.normalize("NFC", texte)
    return texte


def tokeniser(texte: str) -> list[str]:
    """Découpe le texte en tokens (mots) en ne gardant que les séquences
    alphanumériques (y compris les caractères accentués)."""
    return re.findall(r"[\w]+", normaliser_texte(texte), flags=re.UNICODE)


def similarite(mot_a: str, mot_b: str) -> float:
    """Calcule la similarité entre deux mots via SequenceMatcher (ratio 0..1)."""
    return SequenceMatcher(None, mot_a, mot_b).ratio()


# ---------------------------------------------------------------------------
# Extraction de base avec PyMuPDF
# ---------------------------------------------------------------------------

def extraire_texte_pymupdf(chemin_pdf: Path) -> str:
    """Extrait le texte brut d'un PDF via PyMuPDF.

    Gère les PDF corrompus en renvoyant une chaîne vide avec un message d'erreur.
    """
    try:
        doc = fitz.open(str(chemin_pdf))
    except Exception as e:
        print(f"  [ERREUR] Impossible d'ouvrir '{chemin_pdf.name}' : {e}")
        return ""

    texte_complet = []
    for page in doc:
        try:
            texte_complet.append(page.get_text())
        except Exception as e:
            print(f"  [AVERTISSEMENT] Page ignorée dans '{chemin_pdf.name}' : {e}")
    doc.close()
    return "\n".join(texte_complet)


# ---------------------------------------------------------------------------
# Traitement Mistral OCR
# ---------------------------------------------------------------------------

def transcrire_mistral_ocr(client: Mistral, chemin_pdf: Path) -> str:
    """Envoie un PDF à l'API Mistral OCR et retourne la transcription complète.

    Le PDF est encodé en base64 et envoyé via l'endpoint OCR de Mistral.
    """
    # Lecture et encodage du PDF en base64
    contenu_pdf = chemin_pdf.read_bytes()
    pdf_base64 = base64.standard_b64encode(contenu_pdf).decode("utf-8")

    # Appel à l'API Mistral OCR
    try:
        resultat_ocr = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{pdf_base64}",
            },
        )
    except Exception as e:
        print(f"  [ERREUR] Mistral OCR a échoué pour '{chemin_pdf.name}' : {e}")
        return ""

    # Extraction du texte depuis les pages du résultat OCR
    texte_pages = []
    for page in resultat_ocr.pages:
        texte_pages.append(page.markdown)
    return "\n".join(texte_pages)


# ---------------------------------------------------------------------------
# Comparaison et détection des erreurs
# ---------------------------------------------------------------------------

def comparer_tokens(tokens_base: list[str], tokens_ocr: list[str],
                    seuil: float) -> list[str]:
    """Compare les tokens de l'extraction de base avec ceux de Mistral OCR.

    Pour chaque token de l'extraction de base, on cherche le meilleur
    correspondant parmi les tokens OCR dans une fenêtre glissante. Si la
    meilleure similarité est inférieure au seuil, le terme est considéré
    comme 'non reconnu'.

    Retourne la liste des termes non reconnus.
    """
    termes_non_reconnus = []
    ensemble_ocr = set(tokens_ocr)

    # Index pour parcourir les tokens OCR de manière séquentielle
    idx_ocr = 0
    taille_fenetre = 10  # Fenêtre de recherche autour de la position courante

    for token_base in tokens_base:
        # Vérification rapide : le mot existe-t-il exactement dans l'OCR ?
        if token_base in ensemble_ocr:
            # Avancer l'index OCR si possible
            try:
                pos = tokens_ocr.index(token_base, max(0, idx_ocr - taille_fenetre))
                idx_ocr = pos + 1
            except ValueError:
                pass
            continue

        # Recherche dans une fenêtre glissante autour de la position courante
        debut = max(0, idx_ocr - taille_fenetre)
        fin = min(len(tokens_ocr), idx_ocr + taille_fenetre)
        fenetre = tokens_ocr[debut:fin]

        if not fenetre:
            termes_non_reconnus.append(token_base)
            continue

        meilleure_sim = max(similarite(token_base, t) for t in fenetre)

        if meilleure_sim < seuil:
            termes_non_reconnus.append(token_base)
        else:
            # Avancer l'index OCR
            for i, t in enumerate(fenetre):
                if similarite(token_base, t) == meilleure_sim:
                    idx_ocr = debut + i + 1
                    break

    return termes_non_reconnus


# ---------------------------------------------------------------------------
# Traitement principal d'un PDF
# ---------------------------------------------------------------------------

def traiter_pdf(chemin_pdf: Path, racine: Path, client: Mistral,
                seuil: float) -> int:
    """Traite un seul PDF : extraction, OCR Mistral, comparaison, erreurs.

    Retourne le nombre de termes non reconnus.
    """
    nom_base = chemin_pdf.stem  # Nom sans extension

    # --- Étape 1 : Extraction de base (PyMuPDF) ---
    print(f"  [1/4] Extraction de base (PyMuPDF)...")
    texte_base = extraire_texte_pymupdf(chemin_pdf)
    if not texte_base.strip():
        print(f"  [AVERTISSEMENT] Aucun texte extrait de '{chemin_pdf.name}'.")

    # Sauvegarde du texte extrait
    chemin_output = racine / DOSSIER_OUTPUT / f"{nom_base}.txt"
    chemin_output.write_text(texte_base, encoding="utf-8")
    print(f"  [1/4] Sauvegardé → {chemin_output.name}")

    # --- Étape 2 : Mistral OCR ---
    print(f"  [2/4] Transcription via Mistral OCR...")
    texte_ocr = transcrire_mistral_ocr(client, chemin_pdf)
    if not texte_ocr.strip():
        print(f"  [AVERTISSEMENT] Mistral OCR n'a retourné aucun texte pour "
              f"'{chemin_pdf.name}'.")

    # Sauvegarde du résultat OCR
    chemin_ocr = racine / DOSSIER_MISTRAL / f"{nom_base}.txt"
    chemin_ocr.write_text(texte_ocr, encoding="utf-8")
    print(f"  [2/4] Sauvegardé → {chemin_ocr.name}")

    # --- Étape 3 : Tokenisation et comparaison ---
    print(f"  [3/4] Comparaison terme à terme...")
    tokens_base = tokeniser(texte_base)
    tokens_ocr = tokeniser(texte_ocr)

    if not tokens_base:
        print(f"  [AVERTISSEMENT] Aucun token dans l'extraction de base.")
        return 0
    if not tokens_ocr:
        print(f"  [AVERTISSEMENT] Aucun token OCR — tous les termes seront "
              f"considérés comme non reconnus.")
        termes_non_reconnus = tokens_base
    else:
        termes_non_reconnus = comparer_tokens(tokens_base, tokens_ocr, seuil)

    # --- Étape 4 : Journal des erreurs ---
    nb_erreurs = len(termes_non_reconnus)
    print(f"  [4/4] {nb_erreurs} terme(s) non reconnu(s) sur "
          f"{len(tokens_base)} tokens.")

    if nb_erreurs > 0:
        # Le fichier d'erreurs est nommé par le nombre d'erreurs
        dossier_erreurs_pdf = racine / DOSSIER_ERREURS / nom_base
        dossier_erreurs_pdf.mkdir(parents=True, exist_ok=True)
        chemin_erreurs = dossier_erreurs_pdf / f"{nb_erreurs}.txt"

        contenu_erreurs = (
            f"# Termes non reconnus pour : {chemin_pdf.name}\n"
            f"# Nombre total de termes analysés : {len(tokens_base)}\n"
            f"# Nombre de termes non reconnus : {nb_erreurs}\n"
            f"# Seuil de similarité utilisé : {seuil}\n"
            f"# {'=' * 60}\n\n"
        )
        contenu_erreurs += "\n".join(termes_non_reconnus)

        chemin_erreurs.write_text(contenu_erreurs, encoding="utf-8")
        print(f"  [4/4] Erreurs sauvegardées → {chemin_erreurs}")

    return nb_erreurs


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    """Fonction principale : parse les arguments, initialise Mistral et traite
    chaque PDF du dossier source."""

    # --- Arguments en ligne de commande ---
    parser = argparse.ArgumentParser(
        description="Vérification OCR de volumes PDF via Mistral OCR."
    )
    parser.add_argument(
        "--source", type=str, default=DOSSIER_SOURCE,
        help=f"Dossier contenant les PDF source (défaut : {DOSSIER_SOURCE})"
    )
    parser.add_argument(
        "--seuil", type=float, default=SEUIL_SIMILARITE,
        help=f"Seuil de similarité pour la comparaison (défaut : {SEUIL_SIMILARITE})"
    )
    args = parser.parse_args()

    # --- Répertoire racine du projet ---
    racine = Path(__file__).resolve().parent

    # --- Vérification de la clé API ---
    cle_api = os.environ.get("MISTRAL_API_KEY")
    if not cle_api:
        print("[ERREUR] La variable d'environnement MISTRAL_API_KEY n'est pas définie.")
        print("  → export MISTRAL_API_KEY=\"votre_clé_api\"")
        sys.exit(1)

    # --- Initialisation du client Mistral ---
    client = Mistral(api_key=cle_api)

    # --- Création des dossiers ---
    creer_dossiers(racine)

    # --- Collecte des PDF ---
    dossier_source = racine / args.source
    if not dossier_source.exists():
        print(f"[ERREUR] Le dossier source '{dossier_source}' n'existe pas.")
        sys.exit(1)

    fichiers_pdf = sorted(dossier_source.glob("*.pdf"))
    if not fichiers_pdf:
        print(f"[INFO] Aucun fichier PDF trouvé dans '{dossier_source}'.")
        sys.exit(0)

    print(f"[INFO] {len(fichiers_pdf)} fichier(s) PDF trouvé(s) dans '{dossier_source}'.")
    print(f"[INFO] Seuil de similarité : {args.seuil}")
    print(f"{'=' * 70}")

    # --- Traitement de chaque PDF avec barre de progression ---
    total_erreurs = 0
    resultats = {}

    for chemin_pdf in tqdm(fichiers_pdf, desc="Traitement des PDF", unit="PDF"):
        print(f"\n📄 Traitement de : {chemin_pdf.name}")
        nb_erreurs = traiter_pdf(chemin_pdf, racine, client, args.seuil)
        total_erreurs += nb_erreurs
        resultats[chemin_pdf.name] = nb_erreurs

    # --- Résumé final ---
    print(f"\n{'=' * 70}")
    print(f"[RÉSUMÉ] Traitement terminé pour {len(fichiers_pdf)} fichier(s).")
    print(f"[RÉSUMÉ] Total des termes non reconnus : {total_erreurs}")
    print(f"\nDétail par fichier :")
    for nom, nb in resultats.items():
        statut = "✓" if nb == 0 else f"✗ {nb} erreur(s)"
        print(f"  {nom} → {statut}")


if __name__ == "__main__":
    main()
