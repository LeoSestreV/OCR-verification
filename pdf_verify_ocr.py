import os
import re
import sys
import unicodedata
import logging
import warnings
import io
from pathlib import Path
from difflib import SequenceMatcher

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['FLAGS_allocator_strategy'] = 'naive_best_fit'
warnings.filterwarnings("ignore", category=UserWarning, module='requests')

import fitz
import numpy as np
from PIL import Image
import paddle
from paddleocr import PaddleOCR
from tqdm import tqdm

paddle.set_flags({'FLAGS_enable_pir_api': 0})
logging.getLogger("ppocr").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
SEUIL_SIMILARITE = 0.75
LONGUEUR_MIN_MOT = 3  # Ignorer les mots de moins de 3 caractères (à, au, le, etc.)
DOSSIER_SOURCE = "BioPDF"
DOSSIER_OUTPUT = "output_txt"
DOSSIER_OCR = "ocr_paddle"
DOSSIER_ERREURS = "erreurs"
DPI_RESOLUTION = 200


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def creer_dossiers(racine: Path) -> None:
    for nom in [DOSSIER_OUTPUT, DOSSIER_OCR, DOSSIER_ERREURS]:
        (racine / nom).mkdir(parents=True, exist_ok=True)


def supprimer_accents(texte: str) -> str:
    """Supprime tous les accents et diacritiques d'un texte.
    Exemple : 'célèbre' -> 'celebre', 'à' -> 'a'."""
    nfkd = unicodedata.normalize("NFKD", texte)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def tokeniser(texte: str) -> list[str]:
    """Découpe le texte en tokens (mots) avec accents préservés."""
    texte_lower = unicodedata.normalize("NFC", texte.lower())
    return re.findall(r"[\w]+", texte_lower, flags=re.UNICODE)


def tokeniser_sans_accents(texte: str) -> list[str]:
    """Découpe le texte en tokens normalisés sans accents pour la comparaison."""
    texte_norm = supprimer_accents(texte.lower())
    return re.findall(r"[\w]+", texte_norm, flags=re.UNICODE)


def similarite(mot_a: str, mot_b: str) -> float:
    return SequenceMatcher(None, mot_a, mot_b).ratio()


# ---------------------------------------------------------------------------
# Extraction de base avec PyMuPDF — texte continu (pas en colonnes)
# ---------------------------------------------------------------------------

def extraire_texte_pymupdf(chemin_pdf: Path) -> str:
    """Extrait le texte d'un PDF via PyMuPDF en mode texte continu.

    Utilise l'extraction par blocs triés verticalement puis horizontalement
    pour reconstituer un flux de texte continu au lieu de garder la disposition
    en colonnes du PDF original.
    """
    try:
        doc = fitz.open(str(chemin_pdf))
    except Exception as e:
        print(f"  [ERREUR] Impossible d'ouvrir '{chemin_pdf.name}' : {e}")
        return ""
    texte_complet = []
    for page in doc:
        try:
            # Extraction par blocs de texte avec leurs coordonnées
            blocs = page.get_text("blocks")
            # Chaque bloc = (x0, y0, x1, y1, texte, bloc_no, type)
            # On ne garde que les blocs de texte (type 0), pas les images (type 1)
            blocs_texte = [b for b in blocs if b[6] == 0]

            # Tri par position verticale (y0 arrondi) puis horizontale (x0)
            # pour reconstituer l'ordre de lecture naturel
            blocs_texte.sort(key=lambda b: (round(b[1] / 10) * 10, b[0]))

            # Assemblage en texte continu
            texte_page = " ".join(b[4].strip() for b in blocs_texte if b[4].strip())
            texte_complet.append(texte_page)
        except Exception as e:
            print(f"  [AVERTISSEMENT] Page ignorée dans '{chemin_pdf.name}' : {e}")
    doc.close()
    return "\n".join(texte_complet)


# ---------------------------------------------------------------------------
# Traitement PaddleOCR (GPU)
# ---------------------------------------------------------------------------

def transcrire_paddle_ocr(ocr: PaddleOCR, chemin_pdf: Path) -> str:
    try:
        doc = fitz.open(str(chemin_pdf))
    except Exception as e:
        print(f"  [ERREUR] Impossible d'ouvrir '{chemin_pdf.name}' pour OCR : {e}")
        return ""
    texte_complet = []
    for num_page in range(len(doc)):
        try:
            page = doc[num_page]
            matrice = fitz.Matrix(DPI_RESOLUTION / 72, DPI_RESOLUTION / 72)
            pixmap = page.get_pixmap(matrix=matrice)
            img_bytes = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            img_array = np.array(img)
            resultat = ocr.ocr(img_array, cls=True)
            if resultat and resultat[0]:
                lignes_page = [ligne[1][0] for ligne in resultat[0]]
                texte_complet.append(" ".join(lignes_page))
            del pixmap
            img.close()
        except Exception as e:
            print(f"  [AVERTISSEMENT] OCR échoué page {num_page + 1} : {e}")
    doc.close()
    return "\n".join(texte_complet)


# ---------------------------------------------------------------------------
# Comparaison et détection des erreurs (sans accents)
# ---------------------------------------------------------------------------

def comparer_tokens(tokens_base: list[str], tokens_base_norm: list[str],
                    tokens_ocr_norm: list[str], seuil: float,
                    longueur_min: int) -> list[str]:
    """Compare les tokens en version SANS ACCENTS pour éviter les faux positifs
    liés aux différences d'encodage des diacritiques entre PyMuPDF et PaddleOCR.

    Les mots trop courts (< longueur_min) et les nombres purs sont ignorés.

    Retourne la liste des termes non reconnus (dans leur forme originale).
    """
    termes_non_reconnus = []
    ensemble_ocr = set(tokens_ocr_norm)
    idx_ocr = 0
    taille_fenetre = 15

    for i, token_norm in enumerate(tokens_base_norm):
        # Ignorer les mots trop courts
        if len(token_norm) < longueur_min:
            continue

        # Ignorer les tokens purement numériques
        if token_norm.isdigit():
            continue

        # Vérification rapide : le mot (sans accents) existe-t-il dans l'OCR ?
        if token_norm in ensemble_ocr:
            try:
                pos = tokens_ocr_norm.index(token_norm, max(0, idx_ocr - taille_fenetre))
                idx_ocr = pos + 1
            except ValueError:
                pass
            continue

        debut = max(0, idx_ocr - taille_fenetre)
        fin = min(len(tokens_ocr_norm), idx_ocr + taille_fenetre)
        fenetre = tokens_ocr_norm[debut:fin]

        if not fenetre:
            termes_non_reconnus.append(tokens_base[i])
            continue

        meilleure_sim = max(similarite(token_norm, t) for t in fenetre)

        if meilleure_sim < seuil:
            termes_non_reconnus.append(tokens_base[i])
        else:
            for j, t in enumerate(fenetre):
                if similarite(token_norm, t) == meilleure_sim:
                    idx_ocr = debut + j + 1
                    break

    return termes_non_reconnus


# ---------------------------------------------------------------------------
# Traitement principal d'un PDF
# ---------------------------------------------------------------------------

def traiter_pdf(chemin_pdf: Path, racine: Path, ocr: PaddleOCR, seuil: float) -> int:
    nom_base = chemin_pdf.stem

    # Étape 1 : Extraction de base (PyMuPDF, texte continu)
    texte_base = extraire_texte_pymupdf(chemin_pdf)
    (racine / DOSSIER_OUTPUT / f"{nom_base}.txt").write_text(texte_base, encoding="utf-8")

    # Étape 2 : PaddleOCR (GPU)
    print(f"  [OCR GPU] Traitement de {chemin_pdf.name}...")
    texte_ocr = transcrire_paddle_ocr(ocr, chemin_pdf)
    (racine / DOSSIER_OCR / f"{nom_base}.txt").write_text(texte_ocr, encoding="utf-8")

    # Étape 3 : Tokenisation et comparaison (sans accents)
    tokens_base = tokeniser(texte_base)
    tokens_base_norm = tokeniser_sans_accents(texte_base)
    tokens_ocr_norm = tokeniser_sans_accents(texte_ocr)

    if not tokens_base:
        return 0

    termes_non_reconnus = comparer_tokens(
        tokens_base, tokens_base_norm, tokens_ocr_norm,
        seuil, LONGUEUR_MIN_MOT
    )

    # Dédoublonner tout en gardant l'ordre d'apparition
    vus = set()
    termes_uniques = []
    for t in termes_non_reconnus:
        if t not in vus:
            vus.add(t)
            termes_uniques.append(t)

    nb_erreurs = len(termes_non_reconnus)
    nb_uniques = len(termes_uniques)

    # Étape 4 : Journal des erreurs
    if nb_erreurs > 0:
        dossier_erreurs_pdf = racine / DOSSIER_ERREURS / nom_base
        dossier_erreurs_pdf.mkdir(parents=True, exist_ok=True)
        chemin_err = dossier_erreurs_pdf / f"{nb_erreurs}.txt"
        header = (
            f"# Termes non reconnus pour : {chemin_pdf.name}\n"
            f"# Erreurs: {nb_erreurs}/{len(tokens_base)} "
            f"({nb_uniques} termes uniques)\n"
            f"# Seuil: {seuil} | Longueur min: {LONGUEUR_MIN_MOT}\n"
            f"{'=' * 50}\n"
        )
        chemin_err.write_text(header + "\n".join(termes_uniques), encoding="utf-8")

    return nb_erreurs


def main():
    racine = Path(__file__).resolve().parent
    creer_dossiers(racine)
    dossier_source = racine / DOSSIER_SOURCE
    fichiers_pdf = sorted(dossier_source.glob("*.pdf"))
    if not fichiers_pdf:
        print("Aucun PDF trouvé.")
        return
    print(f"[INFO] Initialisation de PaddleOCR sur GPU...")
    print(f"[INFO] Seuil de similarité : {SEUIL_SIMILARITE}")
    print(f"[INFO] Longueur minimale des mots : {LONGUEUR_MIN_MOT}")
    ocr = PaddleOCR(use_angle_cls=True, lang="fr", use_gpu=True)
    resultats = {}
    for chemin_pdf in tqdm(fichiers_pdf, desc="Traitement Global", unit="PDF"):
        nb = traiter_pdf(chemin_pdf, racine, ocr, SEUIL_SIMILARITE)
        resultats[chemin_pdf.name] = nb
    print("\n--- TERMINE ---")
    for k, v in resultats.items():
        print(f"{k} : {v} erreurs")

if __name__ == "__main__":
    main()
