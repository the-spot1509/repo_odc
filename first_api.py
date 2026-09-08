from fastapi import FastAPI 
import random

app = FastAPI()
@app.get("/")

def acceuil():
    return {"message": "Salut moi c'est yvan et voici mon premier API avec FastAPI"}

@app.get("/random")
def get_random_number():
    return {"number": random.randint(1, 100)}

@app.get("/exposant/{x}_{y}")
def fonction_exposant(x: int, y: int):
    """
    cette fonction permet de calculer l'exposant d'un nombre x à la puissance y
    """
    return {f"l'exposant de {x} à la puissance {y} est": x ** y}