#!/usr/bin/env python3
"""
pdf_verify_ocr.py — Script de vérification OCR pour volumes de biographies nationales en PDF.

Ce script :
1. Extrait le texte de chaque PDF via PyMuPDF (extraction de base).
2. Effectue une reconnaissance OCR via PaddleOCR sur chaque page du PDF.
3. Compare les deux résultats terme à terme.
4. Génère un fichier d'erreurs nommé par le nombre de termes non reconnus.

Dossiers utilisés :
    /source_pdf/   — PDF originaux (par défaut : BioPDF/)
    /output_txt/   — Textes issus de l'extraction de base (PyMuPDF)
    /ocr_paddle/   — Résultats complets de PaddleOCR
    /erreurs/      — Fichiers nommés par le nombre d'erreurs

Utilisation :
    python pdf_verify_ocr.py

Dépendances :
    pip install -r requirements.txt
"""

import re
import sys
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
SEUIL_SIMILARITE = 0.75  # En dessous de ce seuil, le terme est considéré non reconnu
DOSSIER_SOURCE = "BioPDF"  # Dossier contenant les PDF originaux
DOSSIER_OUTPUT = "output_txt"
DOSSIER_OCR = "ocr_paddle"
DOSSIER_ERREURS = "erreurs"


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def creer_dossiers(racine: Path) -> None:
    """Crée les dossiers de sortie s'ils n'existent pas."""
    for nom in [DOSSIER_OUTPUT, DOSSIER_OCR, DOSSIER_ERREURS]:
        (racine / nom).mkdir(parents=True, exist_ok=True)


def normaliser_texte(texte: str) -> str:
    """Normalise le texte : minuscules et normalisation Unicode."""
    texte = texte.lower()
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
# Traitement PaddleOCR
# ---------------------------------------------------------------------------

def transcrire_paddle_ocr(ocr: PaddleOCR, chemin_pdf: Path) -> str:
    """Convertit chaque page du PDF en image puis applique PaddleOCR.

    Utilise PyMuPDF pour convertir les pages en images (pixmap) puis
    PaddleOCR pour la reconnaissance de texte sur chaque image.

    Retourne le texte complet reconnu par PaddleOCR.
    """
    try:
        doc = fitz.open(str(chemin_pdf))
    except Exception as e:
        print(f"  [ERREUR] Impossible d'ouvrir '{chemin_pdf.name}' pour OCR : {e}")
        return ""

    texte_complet = []
    for num_page in range(len(doc)):
        try:
            page = doc[num_page]
            # Conversion de la page en image haute résolution (300 DPI)
            matrice = fitz.Matrix(300 / 72, 300 / 72)
            pixmap = page.get_pixmap(matrix=matrice)

            # Sauvegarde temporaire de l'image en mémoire (format PNG en bytes)
            img_bytes = pixmap.tobytes("png")

            # Écriture temporaire sur disque (PaddleOCR nécessite un chemin ou un array numpy)
            import numpy as np
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(img_bytes))
            img_array = np.array(img)

            # Exécution de PaddleOCR sur l'image
            resultat = ocr.ocr(img_array, cls=True)

            # Extraction du texte reconnu
            if resultat and resultat[0]:
                lignes_page = []
                for ligne in resultat[0]:
                    texte_ligne = ligne[1][0]  # (coordonnées, (texte, confiance))
                    lignes_page.append(texte_ligne)
                texte_complet.append("\n".join(lignes_page))

        except Exception as e:
            print(f"  [AVERTISSEMENT] OCR échoué page {num_page + 1} "
                  f"de '{chemin_pdf.name}' : {e}")

    doc.close()
    return "\n".join(texte_complet)


# ---------------------------------------------------------------------------
# Comparaison et détection des erreurs
# ---------------------------------------------------------------------------

def comparer_tokens(tokens_base: list[str], tokens_ocr: list[str],
                    seuil: float) -> list[str]:
    """Compare les tokens de l'extraction de base avec ceux de PaddleOCR.

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

def traiter_pdf(chemin_pdf: Path, racine: Path, ocr: PaddleOCR,
                seuil: float) -> int:
    """Traite un seul PDF : extraction, PaddleOCR, comparaison, erreurs.

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

    # --- Étape 2 : PaddleOCR ---
    print(f"  [2/4] Transcription via PaddleOCR...")
    texte_ocr = transcrire_paddle_ocr(ocr, chemin_pdf)
    if not texte_ocr.strip():
        print(f"  [AVERTISSEMENT] PaddleOCR n'a retourné aucun texte pour "
              f"'{chemin_pdf.name}'.")

    # Sauvegarde du résultat OCR
    chemin_ocr = racine / DOSSIER_OCR / f"{nom_base}.txt"
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
    """Fonction principale : initialise PaddleOCR et traite chaque PDF
    du dossier source."""

    # --- Répertoire racine du projet ---
    racine = Path(__file__).resolve().parent

    # --- Création des dossiers de sortie ---
    creer_dossiers(racine)

    # --- Collecte des PDF ---
    dossier_source = racine / DOSSIER_SOURCE
    if not dossier_source.exists():
        print(f"[ERREUR] Le dossier source '{dossier_source}' n'existe pas.")
        sys.exit(1)

    fichiers_pdf = sorted(dossier_source.glob("*.pdf"))
    if not fichiers_pdf:
        print(f"[INFO] Aucun fichier PDF trouvé dans '{dossier_source}'.")
        sys.exit(0)

    print(f"[INFO] {len(fichiers_pdf)} fichier(s) PDF trouvé(s) dans '{dossier_source}'.")
    print(f"[INFO] Seuil de similarité : {SEUIL_SIMILARITE}")
    print(f"{'=' * 70}")

    # --- Initialisation de PaddleOCR ---
    # lang='fr' pour le français, show_log=False pour éviter le bruit dans la console
    print("[INFO] Initialisation de PaddleOCR (modèle français)...")
    ocr = PaddleOCR(use_angle_cls=True, lang="fr", show_log=False)

    # --- Traitement de chaque PDF avec barre de progression ---
    total_erreurs = 0
    resultats = {}

    for chemin_pdf in tqdm(fichiers_pdf, desc="Traitement des PDF", unit="PDF"):
        print(f"\n Traitement de : {chemin_pdf.name}")
        nb_erreurs = traiter_pdf(chemin_pdf, racine, ocr, SEUIL_SIMILARITE)
        total_erreurs += nb_erreurs
        resultats[chemin_pdf.name] = nb_erreurs

    # --- Résumé final ---
    print(f"\n{'=' * 70}")
    print(f"[RÉSUMÉ] Traitement terminé pour {len(fichiers_pdf)} fichier(s).")
    print(f"[RÉSUMÉ] Total des termes non reconnus : {total_erreurs}")
    print(f"\nDétail par fichier :")
    for nom, nb in resultats.items():
        statut = "OK" if nb == 0 else f"{nb} erreur(s)"
        print(f"  {nom} -> {statut}")


if __name__ == "__main__":
    main()
