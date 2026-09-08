import streamlit as st
import requests

# Configuration de la page
st.set_page_config(
    page_title="Mon API FastAPI",
    page_icon="🚀",
    layout="centered"
)

# URL de ton API FastAPI
API_URL = "http://127.0.0.1:8000"

# -------------------------
# TITRE
# -------------------------

st.title("🚀 Mon API FastAPI")
st.write("Une interface simple pour utiliser mon API.")

st.divider()

# -------------------------
# ACCUEIL
# -------------------------

st.header("👋 Accueil")

if st.button("Afficher le message de l'API"):
    response = requests.get(f"{API_URL}/")

    if response.status_code == 200:
        data = response.json()
        st.success(data["message"])
    else:
        st.error("Impossible de contacter l'API")


# -------------------------
# NOMBRE ALÉATOIRE
# -------------------------

st.divider()

st.header("🎲 Générateur de nombre aléatoire")

if st.button("Générer un nombre"):
    response = requests.get(f"{API_URL}/random")

    if response.status_code == 200:
        data = response.json()

        st.metric(
            label="Nombre généré",
            value=data["number"]
        )
    else:
        st.error("Erreur lors de la récupération du nombre")


# -------------------------
# EXPOSANT
# -------------------------

st.divider()

st.header("🧮 Calcul d'une puissance")

col1, col2 = st.columns(2)

with col1:
    x = st.number_input(
        "Nombre x",
        value=2,
        step=1
    )

with col2:
    y = st.number_input(
        "Puissance y",
        value=3,
        step=1
    )

if st.button("Calculer", type="primary"):

    # Conversion en entier car ton API attend x:int et y:int
    x = int(x)
    y = int(y)

    response = requests.get(
        f"{API_URL}/exposant/{x}_{y}"
    )

    if response.status_code == 200:
        data = response.json()

        resultat = list(data.values())[0]

        st.success(f"Résultat : {resultat}")

    else:
        st.error("Erreur lors du calcul")