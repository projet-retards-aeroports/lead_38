# ✈️ Lead 38 - Retards Aéroports

**Version Lead** du projet d'analyse et de prédiction des retards aériens.

Ce dépôt contient l'architecture complète du projet (code source, notebooks, API, application Streamlit et tracking MLflow).

## 📁 Structure du projet

| Dossier       | Description                                      | Type          |
|---------------|--------------------------------------------------|---------------|
| `src/`        | Code source principal (Python)                   | Code          |
| `notebooks/`  | Notebooks d'exploration et d'analyse             | Analyse       |
| `api/`        | API FastAPI                                      | **Submodule** (Hugging Face) |
| `streamlit/`  | Application web Streamlit                        | **Submodule** (Hugging Face) |
| `mlflow/`     | Expérimentations et tracking MLflow              | **Submodule** (Hugging Face) |

## 🚀 Clonage du projet

```bash
git clone --recurse-submodules https://github.com/projet-retards-aeroports/lead_38.git
cd lead_38


🔗 Submodules Hugging Face
Ce dépôt utilise des Git Submodules pour lier les applications déployées sur Hugging Face :

API → https://huggingface.co/spaces/projetLead38/retards_aero_api
Streamlit → https://huggingface.co/spaces/projetLead38/retards_aero_streamlit
MLflow → https://huggingface.co/spaces/projetLead38/pl_mlflow


🔄 Mettre à jour depuis les dossiers api/, streamlit/ ou mlflow/
Méthode 1 : Pousser directement sur Hugging Face (recommandée)
Bashcd api          # ou streamlit / mlflow
git add .
git commit -m "Description des changements"
git push
Méthode 2 : Mettre à jour le lien dans le repo principal
Bashcd ~/projets_jedha/prod/lead_38     # adapte le chemin si besoin
git add api
git commit -m "Update api submodule"
git push

🛠️ Commandes utiles

























ActionCommandeOù l'exécuterPousser sur GitHubgit add . && git commit -m "msg" && git pushRacinePousser sur Hugging Facegit add . && git commit -m "msg" && git pushDans le submoduleMettre à jour tous les submodulesgit submodule update --remote --mergeRacine

Auteur : projetLead38
Repo Lead : lead_38
text
