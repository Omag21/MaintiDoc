# MaintiDoc SAN

Assistant documentaire pour la maintenance d'une baie de stockage SAN **fictive** (SAN-LAB 2000) : un RAG multimodal local qui répond à partir d'un manuel PDF en citant ses pages et ses figures.

Projet intégrateur — IA générative visuelle et textuelle.

## Fonctionnement

1. **Extraction** du PDF avec PyMuPDF : texte, tableaux et schémas.
2. **Index texte** : LlamaIndex et embeddings BGE-M3, stockés dans Qdrant.
3. **Index image** : embeddings CLIP des schémas, stockés dans Qdrant.
4. **Fusion** des deux recherches par Reciprocal Rank Fusion (RRF), au niveau des pages.
5. **Génération** par un modèle local via Ollama, avec une réponse JSON imposée.
6. **Contrôle** des citations et trace JSON de provenance.

Qdrant tourne en mode local (dossier sur disque) : aucun serveur ni Docker n'est nécessaire.

## Contenu

| Fichier | Rôle |
|---|---|
| `MaintiDoc_SAN_Qwen3_8_JSON_c.ipynb` | Notebook complet, de l'extraction à la réponse vérifiée |
| `app.py` | Application Streamlit de démonstration |
| `Manuel_Maintenance_SAN_Projet_Integrateur.pdf` | Manuel fictif utilisé comme corpus |
| `requirements.txt` | Dépendances Python |
| `maintidoc-san/` | Projet `uv` (pyproject.toml et uv.lock) |
| `*.png` | Schémas d'architecture et feuille de route |

## Installation

Python 3.14 ou plus récent est requis.

Avec pip :

```bash
python -m venv .venv
pip install -r requirements.txt
```

Ou avec uv :

```bash
cd maintidoc-san
uv sync
```

Installer ensuite [Ollama](https://ollama.com/download) et télécharger un modèle :

```bash
ollama pull qwen3.8:27b
```

`qwen3.8:27b` demande environ 18 Go de mémoire (cible : Mac Apple Silicon 36 Go). Sur une machine plus modeste, choisir un modèle plus petit et modifier `OLLAMA_MODEL` dans le notebook, ou le champ « Modèle Ollama » de l'application.

## Utilisation

**Notebook** : placer le PDF à côté du notebook, démarrer Ollama, puis exécuter toutes les cellules dans l'ordre.

**Application** :

```bash
streamlit run app.py
```

Les fichiers générés (`maintidoc_san_extraction/`, `data/`) ne sont pas versionnés : ils sont recréés à l'exécution. BGE-M3 et CLIP sont téléchargés depuis Hugging Face au premier lancement.

## Limites

- Le manuel est fictif : les réponses ne sont pas des instructions de maintenance pour une baie réelle.
- La validation vérifie que les citations existent dans les preuves transmises, pas qu'elles soutiennent réellement la réponse.
- Le mode local de Qdrant n'autorise qu'un processus à la fois par dossier ; pour un accès partagé, utiliser un serveur Qdrant.
- Aucune action sur des équipements ni création de ticket.
