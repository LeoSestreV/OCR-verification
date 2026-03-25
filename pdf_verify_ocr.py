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

SEUIL_SIMILARITE = 0.75 
DOSSIER_SOURCE = "BioPDF"
DOSSIER_OUTPUT = "output_txt"
DOSSIER_OCR = "ocr_paddle"
DOSSIER_ERREURS = "erreurs"
DPI_RESOLUTION = 200  

def creer_dossiers(racine: Path) -> None:
    for nom in [DOSSIER_OUTPUT, DOSSIER_OCR, DOSSIER_ERREURS]:
        (racine / nom).mkdir(parents=True, exist_ok=True)

def normaliser_texte(texte: str) -> str:
    texte = texte.lower()
    texte = unicodedata.normalize("NFC", texte)
    return texte

def tokeniser(texte: str) -> list[str]:
    return re.findall(r"[\w]+", normaliser_texte(texte), flags=re.UNICODE)

def similarite(mot_a: str, mot_b: str) -> float:
    return SequenceMatcher(None, mot_a, mot_b).ratio()

def extraire_texte_pymupdf(chemin_pdf: Path) -> str:
    try:
        doc = fitz.open(str(chemin_pdf))
    except Exception as e:
        print(f"  [ERREUR] Impossible d'ouvrir '{chemin_pdf.name}' : {e}")
        return ""
    texte_complet = []
    for page in doc:
        texte_complet.append(page.get_text())
    doc.close()
    return "\n".join(texte_complet)

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
                texte_complet.append("\n".join(lignes_page))
            del pixmap
            img.close()
        except Exception as e:
            print(f"  [AVERTISSEMENT] OCR échoué page {num_page + 1} : {e}")
    doc.close()
    return "\n".join(texte_complet)

def comparer_tokens(tokens_base: list[str], tokens_ocr: list[str], seuil: float) -> list[str]:
    termes_non_reconnus = []
    ensemble_ocr = set(tokens_ocr)
    idx_ocr = 0
    taille_fenetre = 15 
    for token_base in tokens_base:
        if token_base in ensemble_ocr:
            try:
                pos = tokens_ocr.index(token_base, max(0, idx_ocr - 5))
                idx_ocr = pos + 1
            except ValueError: pass
            continue
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
            for i, t in enumerate(fenetre):
                if similarite(token_base, t) == meilleure_sim:
                    idx_ocr = debut + i + 1
                    break
    return termes_non_reconnus

def traiter_pdf(chemin_pdf: Path, racine: Path, ocr: PaddleOCR, seuil: float) -> int:
    nom_base = chemin_pdf.stem
    texte_base = extraire_texte_pymupdf(chemin_pdf)
    (racine / DOSSIER_OUTPUT / f"{nom_base}.txt").write_text(texte_base, encoding="utf-8")
    print(f"  [OCR GPU] Traitement de {chemin_pdf.name}...")
    texte_ocr = transcrire_paddle_ocr(ocr, chemin_pdf)
    (racine / DOSSIER_OCR / f"{nom_base}.txt").write_text(texte_ocr, encoding="utf-8")
    tokens_base = tokeniser(texte_base)
    tokens_ocr = tokeniser(texte_ocr)
    if not tokens_base: return 0
    termes_non_reconnus = comparer_tokens(tokens_base, tokens_ocr, seuil)
    nb_erreurs = len(termes_non_reconnus)
    if nb_erreurs > 0:
        dossier_erreurs_pdf = racine / DOSSIER_ERREURS / nom_base
        dossier_erreurs_pdf.mkdir(parents=True, exist_ok=True)
        chemin_err = dossier_erreurs_pdf / f"{nb_erreurs}.txt"
        header = f"# Erreurs: {nb_erreurs}/{len(tokens_base)}\n# Seuil: {seuil}\n" + "="*30 + "\n"
        chemin_err.write_text(header + "\n".join(termes_non_reconnus), encoding="utf-8")
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