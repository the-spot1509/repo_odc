#!/usr/bin/env python
# coding: utf-8

# # Appelle des Bibliothèques

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Literal
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# In[5]:


DATASET_PATH = Path(__file__).with_name("dataset_transactions_fraude_5000.csv")
df= pd.read_csv(DATASET_PATH)
df.head()


# # Début de l'EDA

# - Exploration du dataset avec:
#  
#    - Shape pour voir la taille du dataset
#    - Dtypes pour voir les types de nos variables
#    - ET isnull() pour connaitre le nombre de valeurs manquante  

# In[6]:


df.shape


# In[7]:


df.dtypes


# In[8]:


missing = df.isnull().mean()*100
missing[missing > 0].sort_values(ascending= False)  


# ### Détermination des outliers
# 
# ici nous avons utilisé l'IQR(l'écart interquartile) qui est égale a :
#                **Q3** - **Q1**
#                
# Puis nous avons utiliser l'IQR pour calculer les outliers grace a cette formule:
# borne inf= **Q1** - 1.5 * IQR
# borne sup= **Q3** + 1.5 * IQR

# In[9]:


#det des outliers
q1= df[["amount","customer_transaction_count","account_age_days"]].quantile(0.25)
q3=df[["amount","customer_transaction_count","account_age_days"]].quantile(0.75)
iqr= q3 - q1  #determination de l'écart interquartiles
borne_inf= q1 - 1.5 * iqr
borne_sup= q3 + 1.5 * iqr

#enregistrement des outliers
amount_outliers=  df[(df['amount'] < borne_inf['amount']) | (df['amount'] > borne_sup['amount'])]["amount"]

customer_transaction_outliers= df[(df['customer_transaction_count'] < borne_inf['customer_transaction_count'])\
| (df['customer_transaction_count'] > borne_sup['customer_transaction_count'])]["customer_transaction_count"]

account_age_outliers=  df[(df['account_age_days'] < borne_inf['account_age_days']) | (df['account_age_days'] > borne_sup['account_age_days'])]\
["account_age_days"]


# ### Remplacement des valeur manquante dans nos var qui en possède
# 
# Toute les valeurs textuelle manquante on été remplacer par *la valeurs qui ce répète le plus(**le mode**)* et les numérique par la **médiane**

# In[10]:


# 1. Remplacement pour les variables textuelles
df['merchant_category'] = df['merchant_category'].fillna(df['merchant_category'].mode()[0])
df['payment_method'] = df['payment_method'].fillna(df['payment_method'].mode()[0])

# 2. Remplacement pour la variable numérique (par la médiane)
mediane_age = df['customer_age'].median()
df['customer_age'] = df['customer_age'].fillna(mediane_age)


# # Visualisation des données

# ### Repartion d'is_fraud 
# 
# On peut observé via le graphique que notre variable cible a un gros déséquilibre des classe qui se traduit par une importante présence de la valeur 0 représantant les **Transaction normales** , par contre la valeurs 1 qui représante **les  transaction frauduleuse** est casi inexistante. On a ainsi:
# 
#                                                      0 = 97,28% et 1 = 2,72%

# In[11]:


# The API does not need to render notebook charts during startup.
repartition= df['is_fraud'].value_counts()
proportion= df['is_fraud'].value_counts(normalize= True) * 100
df_repartition= pd.DataFrame({'Nombre': repartition , 'Pourcentage(%)': proportion}) 
print(df_repartition)


# ### Analyse du Paire_plot
# 
# normal = bleus , fraude = orange
# 
# 1. En regardant la diagonale qui est uniquement constitué de courbe de densité on remarque que les courbe Orange on globalement la même forme que les Bleues, ce qui nous laisse déja présager une difficulté a présire les Fraude.
# 
# 2. Ensuite viens les Scatter plot du graphique qui confimre la diffuculté de repéré les fraude , car elle sont mélangé/groupé avec les autre

# In[12]:




# ### Analyse de la heat map
# 
# la heat map est utilisé pour observé la corrélation entre divérante variable don jeux de donnée.
# En regardant le notre on voit que toute les variable on une faible corrélation avec notre variable cible 

# In[13]:




# # Phase d'entrainement des models

# In[14]:


#préparation des données
df = pd.get_dummies(df, columns=["merchant_category","customer_country","payment_method"], drop_first = True)


# In[15]:


X= df.drop(['is_fraud','transaction_id'], axis=1)
y= df['is_fraud']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y, random_state=15, stratify=y, test_size=0.2)     


# In[16]:


df.head()


# ## Régression Logistique
# 
# Après avoir préparé les donnée, j'utilise SKlearn pour appellé les model et les outils pour l'optimisé et l'évalué.
# 
# 
# La regression logistique donne un recall de 89% contre une précision de 3% pour un seuille fixé a *0.15* 
# 
# Ce qui signifie qu'après entrainement mon model est capable de détecter 89% des fraude réel , par contre les fraudes réel qu'il détecte ne représente  que 3% de ces prédiction , donc près de 97% de ces prédiction sont fausses

# In[17]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    recall_score, precision_score,
    accuracy_score, f1_score,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

# Standardisation
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèle
m_logistic = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

m_logistic.fit(X_train_scaled, y_train)

# Probabilités prédites par la régression logistique
seuil_choisi = 0.15

y_proba_l = m_logistic.predict_proba(X_test_scaled)[:, 1]

# Application du seuil
y_pred = (y_proba_l >= seuil_choisi).astype(int)

# Évaluation
print(confusion_matrix(y_test, y_pred))

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Fraude"]
    )
)



# ## Random Forest
# 
# - Ce model a été mon second choix, après utilisation de **la validation croisée** sur le **f2_score** j'ai pus trouvéles meilleur paramètre pour mon modèle parmi ceux en liste
# 
# - Ensuite j'ai utilisé une boucle *For* pour essayer de trouver les seuill qui maximise le recall de mon model (voir '**df_resultats_rf**')
# 
# - Ainsi grace au étape décrite plus haut j'ai obtenus un model qui me donne exactement 

# In[18]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, fbeta_score,make_scorer


m_rf = RandomForestClassifier(
    class_weight={0: 1, 1: 30},
    max_depth=6,
    min_samples_leaf=10,
    n_estimators=100
)

m_rf.fit(X_train,y_train)
seuil_choisi_rf = 0.22
y_proba_rf = m_rf.predict_proba(X_test)[:, 1]
y_pred_rf = (y_proba_rf >= seuil_choisi_rf).astype(int)


print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=["Normal", "Fraude"]))


# In[19]:


print(m_rf)


# In[20]:





# ## Test des modèles sur de nouvlle donnée
# 
# La fonction si dessous (**predire_transaction**) permettra d'évalué mes models sur de nouvelle donné.
# 
#           Résultat: Régression logistic indique que ces nouvel donnée sont a 100% des fraude tandis que le Random forest indique 46%  

# In[21]:




# # Analyse avec SHAP
# l'utilisation de SHAP a parmie de savoir quelle variable pesait le plus dans les prédiction de chaque model, ainsi nous pouvons comprendre les résultat précédent 
# 
# - Le random forest ce base en grande partie sur l'heure des transaction pour identifié les transaction frauduleuse.
# - La regression logistique quant a elle ce concentre beacoup sur sur le pays du client masi attention contrairement au random forest d'autre variable comme *la méthode de payment* et l'heure de la transaction son non négligabe et on également un poids concéquant dans 

# In[24]:


app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field = error.get("loc", ["champ"])[-1]
        if field == "is_international":
            message = "is_international doit être égal à 0 ou à 1."
        else:
            message = f"La valeur du champ '{field}' est incorrecte."
        errors.append(message)

    return JSONResponse(
        status_code=422,
        content={"message": "Erreur dans les données saisies.", "errors": errors},
    )

class Transaction(BaseModel):
    amount: float
    transaction_hour: int
    customer_age: int
    is_international: Literal[0, 1]
    customer_transaction_count: int
    account_age_days: int


def prepare_transaction(transaction: Transaction):
    values = transaction.dict()
    row = pd.DataFrame([values])
    row = row.reindex(columns=X_train.columns, fill_value=0)
    return row

@app.get("/")
def read_root():
        return {"message": "Bienvenue sur l'API de détection de fraude"}

@app.post("/predict_logistic")
def predict(transaction: Transaction):
    """
    Cette fonction prend en entrée une transaction et renvoie la prédiction de fraude ainsi que la probbabilité associée en utilisant le modèle de régression logistique.
    veuillez entréz les valeurs de la transaction dans le format suivant:
    {
        "amount": int,
        "transaction_hour": int,
        "customer_age": int,
        "is_international": int,
        "customer_transaction_count": int,
        "account_age_days": int
    }
    """
    x = scaler.transform(prepare_transaction(transaction))
    prediction = int(m_logistic.predict(x)[0])
    proba_fraude = float(m_logistic.predict_proba(x)[0, 1])
    return {
        "prediction": prediction,
        "probability_of_fraud": proba_fraude,
        "message": f"Attention! Cette transaction est suspectée d'être une fraude. la probabilité est de {proba_fraude:.2f}" if prediction == 1 else "Cette transaction semble être normale."
    }    

@app.post("/predict_rf")
def predict_rf(transaction: Transaction):
    """
    Cette fonction prend en entrée une transaction et renvoie la prédiction de fraude ainsi que la probabilité associée en utilisant le modèle Random Forest.
    veuillez entréz les valeurs de la transaction dans le format suivant:
    {
        "amount": int,
        "transaction_hour": int,
        "customer_age": int,
        "is_international": int,
        "customer_transaction_count": int,
        "account_age_days": int
    }
    """
    x = prepare_transaction(transaction)
    prediction = int(m_rf.predict(x)[0])
    proba_fraude = float(m_rf.predict_proba(x)[0, 1])
    return {
        "prediction": prediction,
        "probability_of_fraud": proba_fraude,
        "message": f"Attention! Cette transaction est suspectée d'être une fraude. la probabilité est de {proba_fraude:.2f}" if prediction == 1 else "Cette transaction semble être normale."
    }

