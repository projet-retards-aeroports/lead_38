# ✈️ Projet Retards Aéroports

Projet full-stack d'analyse et de prédiction des retards aériens.

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
git clone --recurse-submodules https://github.com/projet-retards-aeroports/full_stack.git
cd full_stack



🔗 Submodules Hugging Face
Ce dépôt utilise des Git Submodules pour lier les applications déployées sur Hugging Face :

API → https://huggingface.co/spaces/projetLead38/retards_aero_api
Streamlit → https://huggingface.co/spaces/projetLead38/retards_aero_streamlit
MLflow → https://huggingface.co/spaces/projetLead38/pl_mlflow

🔄 Mettre à jour depuis les dossiers api/, streamlit/ ou mlflow/
Méthode 1 : Pousser directement sur Hugging Face (recommandée)
Bash# Aller dans le dossier concerné
cd api          # ou streamlit / mlflow

# Faire tes modifications, puis :
git add .
git commit -m "Description des changements"
git push
Méthode 2 : Mettre à jour le lien dans le repo GitHub principal
Après avoir poussé sur Hugging Face, mets à jour le pointeur du submodule :
Bash# Revenir à la racine
cd ~/projets_jedha/prod/retards_aero

git add api          # (ou streamlit / mlflow)
git commit -m "Update api submodule"
git push

🛠️ Commandes utiles






























ActionCommandeOù l'exécuterPousser sur GitHubgit add . && git commit -m "msg" && git pushRacinePousser sur Hugging Facegit add . && git commit -m "msg" && git pushDans le submoduleMettre à jour tous les submodulesgit submodule update --remote --mergeRacineRe-cloner avec submodulesgit clone --recurse-submodules <url>—

Auteur : projetLead38
