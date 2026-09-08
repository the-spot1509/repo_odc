"""
API de détection de transactions frauduleuses.

Lancer en local :
    uvicorn main:app --reload

Documentation interactive générée automatiquement :
    http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# ------------------------------------------------------------------
# Chargement du modèle et des métadonnées AU DÉMARRAGE de l'API
# (une seule fois, pas à chaque requête -> rapide)
# ------------------------------------------------------------------
model = joblib.load("model/random_forest_fraude.joblib")
metadata = joblib.load("model/metadata.joblib")

COLONNES_X = metadata["colonnes_X"]
COLONNES_CATEGORIELLES = metadata["colonnes_categorielles"]
SEUIL = metadata["seuil"]

app = FastAPI(
    title="API de détection de fraude",
    description="Reçoit les données d'une transaction et retourne une prédiction de fraude.",
    version="1.0",
)


# ------------------------------------------------------------------
# Format des données attendues en entrée (validation automatique)
# ------------------------------------------------------------------
class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Montant de la transaction")
    transaction_hour: int = Field(..., ge=0, le=23, description="Heure de la transaction (0-23)")
    customer_age: float = Field(..., gt=0, description="Âge du client")
    is_international: int = Field(..., ge=0, le=1, description="1 si internationale, 0 sinon")
    customer_transaction_count: int = Field(..., ge=0, description="Nombre de transactions du client")
    account_age_days: int = Field(..., ge=0, description="Ancienneté du compte en jours")
    merchant_category: str = Field(..., description="Catégorie du commerçant")
    customer_country: str = Field(..., description="Pays du client")
    payment_method: str = Field(..., description="Moyen de paiement")

    class Config:
        json_schema_extra = {
            "example": {
                "amount": 750000,
                "transaction_hour": 2,
                "customer_age": 23,
                "is_international": 1,
                "customer_transaction_count": 1,
                "account_age_days": 20,
                "merchant_category": "gaming",
                "customer_country": "Nigeria",
                "payment_method": "wallet",
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    probabilite_fraude: float
    seuil_utilise: float


# ------------------------------------------------------------------
# Fonction de prétraitement, identique à celle du notebook
# ------------------------------------------------------------------
def preparer_transaction(transaction: Transaction) -> pd.DataFrame:
    df_t = pd.DataFrame([transaction.model_dump()])
    df_t = pd.get_dummies(df_t, columns=COLONNES_CATEGORIELLES)
    df_t = df_t.reindex(columns=COLONNES_X, fill_value=0)
    return df_t


# ------------------------------------------------------------------
# Routes de l'API
# ------------------------------------------------------------------
@app.get("/")
def accueil():
    return {"message": "API de détection de fraude opérationnelle. Voir /docs pour tester."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    try:
        df_t = preparer_transaction(transaction)
        proba = float(model.predict_proba(df_t)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de traitement : {e}")

    prediction = "FRAUDE PROBABLE" if proba >= SEUIL else "TRANSACTION NORMALE"

    return PredictionResponse(
        prediction=prediction,
        probabilite_fraude=round(proba, 4),
        seuil_utilise=SEUIL,
    )
