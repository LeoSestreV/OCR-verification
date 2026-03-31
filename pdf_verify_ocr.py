"""
pdf_verify_ocr.py — Vérification OCR de documents PDF via Mistral OCR.

Compare l'extraction textuelle embarquée (PyMuPDF) avec une reconnaissance
optique (Mistral OCR) pour identifier les termes mal reconnus.

Usage :
    python pdf_verify_ocr.py
    python pdf_verify_ocr.py --source dossier/
    python pdf_verify_ocr.py --limit 3 --seuil 0.80
    python pdf_verify_ocr.py --api-key sk-xxx
"""

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from difflib import SequenceMatcher
from typing import Optional

import fitz  # PyMuPDF
from mistralai.client import Mistral
from tqdm import tqdm


def charger_env(chemin: Path) -> None:
    """Charge les variables depuis un fichier .env (clé=valeur)."""
    if not chemin.exists():
        return
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne or ligne.startswith("#") or "=" not in ligne:
            continue
        cle, valeur = ligne.split("=", 1)
        os.environ.setdefault(cle.strip(), valeur.strip())

logger = logging.getLogger("pdf_verify_ocr")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Configuration centralisée — aucune valeur hardcodée dans le code."""
    source: str = "BioPDF"
    output_txt: str = "output_txt"
    output_ocr: str = "ocr_mistral"
    output_erreurs: str = "erreurs"
    seuil: float = 0.75
    taille_fenetre: int = 15
    modele_ocr: str = "mistral-ocr-latest"
    api_key: Optional[str] = None
    limit: Optional[int] = None
    verbose: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Construit une Config depuis les arguments CLI."""
        return cls(
            source=args.source,
            output_txt=args.output_txt,
            output_ocr=args.output_ocr,
            output_erreurs=args.output_erreurs,
            seuil=args.seuil,
            taille_fenetre=args.fenetre,
            modele_ocr=args.modele,
            api_key=args.api_key,
            limit=args.limit,
            verbose=args.verbose,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Normalisation de texte
# ═══════════════════════════════════════════════════════════════════════════

def supprimer_accents(texte: str) -> str:
    """Supprime les diacritiques : 'célèbre' -> 'celebre'."""
    nfkd = unicodedata.normalize("NFKD", texte)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


_RE_MOTS = re.compile(r"[\w]+", re.UNICODE)
_RE_MARKDOWN = re.compile(r"!\[.*?\]\(.*?\)")  # images markdown ![alt](url)


def texte_continu(texte: str) -> str:
    """Convertit un texte (brut ou markdown) en flux continu sans sauts de ligne."""
    texte = _RE_MARKDOWN.sub("", texte)           # retirer les images markdown
    texte = re.sub(r"^#+\s*", "", texte, flags=re.MULTILINE)  # retirer les titres #
    texte = re.sub(r"\n+", " ", texte)             # sauts de ligne -> espaces
    texte = re.sub(r" {2,}", " ", texte)           # espaces multiples -> un seul
    return texte.strip()


def tokeniser(texte: str) -> list[str]:
    """Tokens avec accents préservés (pour le rapport final)."""
    return _RE_MOTS.findall(unicodedata.normalize("NFC", texte.lower()))


def tokeniser_normalise(texte: str) -> list[str]:
    """Tokens sans accents (pour la comparaison)."""
    return _RE_MOTS.findall(supprimer_accents(texte.lower()))


# ═══════════════════════════════════════════════════════════════════════════
# Extraction PyMuPDF — texte continu (anti-colonnes)
# ═══════════════════════════════════════════════════════════════════════════

def extraire_texte(chemin_pdf: Path) -> str:
    """Extrait le texte intégré au PDF en réordonnant les blocs pour
    produire un flux continu (évite le découpage en colonnes)."""
    try:
        doc = fitz.open(str(chemin_pdf))
    except Exception as exc:
        logger.error("Impossible d'ouvrir '%s' : %s", chemin_pdf.name, exc)
        return ""

    pages = []
    for page in doc:
        try:
            blocs = page.get_text("blocks")
            blocs_texte = [b for b in blocs if b[6] == 0]
            blocs_texte.sort(key=lambda b: (round(b[1] / 10) * 10, b[0]))
            pages.append(
                " ".join(b[4].strip() for b in blocs_texte if b[4].strip())
            )
        except Exception as exc:
            logger.warning("Page ignorée dans '%s' : %s", chemin_pdf.name, exc)

    doc.close()
    return texte_continu("\n".join(pages))


# ═══════════════════════════════════════════════════════════════════════════
# OCR via Mistral OCR
# ═══════════════════════════════════════════════════════════════════════════

def ocr_pdf(client: Mistral, chemin_pdf: Path, modele: str) -> str:
    """Envoie le PDF à l'API Mistral OCR et retourne le texte reconnu."""
    pdf_bytes = chemin_pdf.read_bytes()
    base64_pdf = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    try:
        response = client.ocr.process(
            model=modele,
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}",
            },
        )
    except Exception as exc:
        logger.error("Mistral OCR échoué pour '%s' : %s", chemin_pdf.name, exc)
        return ""

    return texte_continu("\n".join(page.markdown for page in response.pages))


# ═══════════════════════════════════════════════════════════════════════════
# Comparaison des tokens
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ResultatComparaison:
    """Résultat de la comparaison entre extraction de base et OCR."""
    total_tokens: int = 0
    nb_erreurs: int = 0
    nb_uniques: int = 0
    termes_uniques: list[str] = field(default_factory=list)


def comparer(texte_base: str, texte_ocr: str, cfg: Config) -> ResultatComparaison:
    """Compare tous les tokens (sans accents) et retourne les termes non reconnus."""
    tokens_orig = tokeniser(texte_base)
    tokens_ocr = tokeniser_normalise(texte_ocr)

    if not tokens_orig:
        return ResultatComparaison()

    ensemble_ocr = set(tokens_ocr)
    idx_ocr = 0
    erreurs = []

    for mot_orig in tokens_orig:
        tok = supprimer_accents(mot_orig)
        if tok.isdigit():
            continue

        # Recherche exacte (rapide)
        if tok in ensemble_ocr:
            try:
                pos = tokens_ocr.index(tok, max(0, idx_ocr - cfg.taille_fenetre))
                idx_ocr = pos + 1
            except ValueError:
                pass
            continue

        # Recherche floue dans une fenêtre
        debut = max(0, idx_ocr - cfg.taille_fenetre)
        fin = min(len(tokens_ocr), idx_ocr + cfg.taille_fenetre)
        fenetre = tokens_ocr[debut:fin]

        if not fenetre:
            erreurs.append(mot_orig)
            continue

        meilleure = max(SequenceMatcher(None, tok, t).ratio() for t in fenetre)

        if meilleure < cfg.seuil:
            erreurs.append(mot_orig)
        else:
            for j, t in enumerate(fenetre):
                if SequenceMatcher(None, tok, t).ratio() == meilleure:
                    idx_ocr = debut + j + 1
                    break

    # Dédoublonner en gardant l'ordre
    vus = set()
    uniques = []
    for t in erreurs:
        if t not in vus:
            vus.add(t)
            uniques.append(t)

    return ResultatComparaison(
        total_tokens=len(tokens_orig),
        nb_erreurs=len(erreurs),
        nb_uniques=len(uniques),
        termes_uniques=uniques,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Traitement d'un fichier PDF
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ResultatPDF:
    """Résultat du traitement complet d'un PDF."""
    fichier: str
    total_tokens: int
    nb_erreurs: int
    nb_uniques: int
    duree_sec: float


def traiter_pdf(chemin_pdf: Path, racine: Path, client: Mistral,
                cfg: Config) -> ResultatPDF:
    """Pipeline complet : extraction -> OCR -> comparaison -> rapport."""
    debut = time.time()
    nom = chemin_pdf.stem

    # 1. Extraction texte intégré (PyMuPDF)
    texte_base = extraire_texte(chemin_pdf)
    (racine / cfg.output_txt / f"{nom}.txt").write_text(texte_base, encoding="utf-8")

    # 2. OCR Mistral
    logger.info("  OCR Mistral de %s...", chemin_pdf.name)
    texte_ocr = ocr_pdf(client, chemin_pdf, cfg.modele_ocr)
    (racine / cfg.output_ocr / f"{nom}.txt").write_text(texte_ocr, encoding="utf-8")

    # 3. Comparaison
    res = comparer(texte_base, texte_ocr, cfg)

    # 4. Fichier d'erreurs
    if res.nb_erreurs > 0:
        dossier = racine / cfg.output_erreurs / nom
        dossier.mkdir(parents=True, exist_ok=True)
        chemin_err = dossier / f"{res.nb_erreurs}.txt"

        en_tete = (
            f"# Fichier : {chemin_pdf.name}\n"
            f"# Tokens analysés : {res.total_tokens}\n"
            f"# Occurrences non reconnues : {res.nb_erreurs}\n"
            f"# Termes uniques non reconnus : {res.nb_uniques}\n"
            f"# Seuil : {cfg.seuil} | Modèle : {cfg.modele_ocr}\n"
            f"{'=' * 50}\n"
        )
        chemin_err.write_text(en_tete + "\n".join(res.termes_uniques), encoding="utf-8")

    duree = time.time() - debut
    return ResultatPDF(
        fichier=chemin_pdf.name,
        total_tokens=res.total_tokens,
        nb_erreurs=res.nb_erreurs,
        nb_uniques=res.nb_uniques,
        duree_sec=round(duree, 1),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Vérification OCR de documents PDF via Mistral OCR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default="BioPDF",
                    help="Dossier contenant les PDF source")
    p.add_argument("--output-txt", default="output_txt",
                    help="Dossier de sortie pour l'extraction textuelle")
    p.add_argument("--output-ocr", default="ocr_mistral",
                    help="Dossier de sortie pour les résultats OCR")
    p.add_argument("--output-erreurs", default="erreurs",
                    help="Dossier de sortie pour les rapports d'erreurs")
    p.add_argument("--seuil", type=float, default=0.75,
                    help="Seuil de similarité (0.0 à 1.0)")
    p.add_argument("--fenetre", type=int, default=15,
                    help="Taille de la fenêtre glissante de comparaison")
    p.add_argument("--modele", default="mistral-ocr-latest",
                    help="Modèle Mistral OCR à utiliser")
    p.add_argument("--api-key", default=None,
                    help="Clé API Mistral (sinon utilise MISTRAL_API_KEY)")
    p.add_argument("--limit", type=int, default=None,
                    help="Limiter le traitement à N fichiers PDF")
    p.add_argument("--verbose", "-v", action="store_true",
                    help="Activer les logs détaillés")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = Config.from_args(args)

    logging.basicConfig(
        level=logging.DEBUG if cfg.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    racine = Path(__file__).resolve().parent

    # Création des dossiers de sortie
    for d in [cfg.output_txt, cfg.output_ocr, cfg.output_erreurs]:
        (racine / d).mkdir(parents=True, exist_ok=True)

    # Collecte des PDF
    dossier_source = racine / cfg.source
    if not dossier_source.exists():
        logger.error("Le dossier source '%s' n'existe pas.", dossier_source)
        sys.exit(1)

    fichiers = sorted(dossier_source.glob("*.pdf"))
    if not fichiers:
        logger.info("Aucun PDF trouvé dans '%s'.", dossier_source)
        sys.exit(0)

    if cfg.limit:
        fichiers = fichiers[:cfg.limit]

    logger.info("%d PDF à traiter | seuil=%.2f | modèle=%s",
                len(fichiers), cfg.seuil, cfg.modele_ocr)

    # Chargement du fichier .env
    charger_env(racine / ".env")

    # Initialisation client Mistral
    api_key = cfg.api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        logger.error("Clé API Mistral requise. Utilisez --api-key ou MISTRAL_API_KEY.")
        sys.exit(1)
    client = Mistral(api_key=api_key)

    # Traitement
    resultats: list[ResultatPDF] = []
    for pdf in tqdm(fichiers, desc="Traitement", unit="PDF"):
        res = traiter_pdf(pdf, racine, client, cfg)
        resultats.append(res)
        logger.info("  %s : %d erreurs (%d uniques) en %.1fs",
                     res.fichier, res.nb_erreurs, res.nb_uniques, res.duree_sec)

    # Résumé
    total_err = sum(r.nb_erreurs for r in resultats)
    total_tok = sum(r.total_tokens for r in resultats)
    duree_tot = sum(r.duree_sec for r in resultats)

    print(f"\n{'=' * 60}")
    print(f"{'FICHIER':<45} {'ERREURS':>8} {'UNIQUES':>8}")
    print(f"{'-' * 60}")
    for r in resultats:
        print(f"{r.fichier:<45} {r.nb_erreurs:>8} {r.nb_uniques:>8}")
    print(f"{'-' * 60}")
    print(f"{'TOTAL':<45} {total_err:>8}")
    print(f"Tokens analysés : {total_tok} | Durée : {duree_tot:.0f}s")

    # Sauvegarde du résumé JSON
    resume = {
        "config": {k: v for k, v in asdict(cfg).items() if k != "api_key"},
        "resultats": [asdict(r) for r in resultats],
        "total_erreurs": total_err,
        "total_tokens": total_tok,
        "duree_totale_sec": duree_tot,
    }
    chemin_resume = racine / cfg.output_erreurs / "resume.json"
    chemin_resume.write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Résumé JSON sauvegardé dans %s", chemin_resume)


if __name__ == "__main__":
    main()
