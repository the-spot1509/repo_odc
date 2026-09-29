import os
import openai  # Indispensable pour intercepter openai.APITimeoutError, etc.
from openai import OpenAI
from dotenv import load_dotenv

# Charge la variable d'environnement (si tu utilises un fichier .env)
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
 
m = "quelle est le role des comcombres dans l'écosystème marin ?"
m2 = "combien font 1 + 1 ?"
r2 = "désolé, je ne peux pas répondre à cette question car elle n'est pas liée aux animaux ou à l'écosystème marin. Si tu as d'autres questions sur les animaux ou l'environnement, je serais ravi de t'aider !"

config = [
    {"role": "system", "content": "Tu es un zoologue expert qui connaît tout ce qu'il faut savoir sur les animaux. Si on te pose une question qui n'est pas liée à la zoologie, tu dois répondre que tu ne peux pas répondre à cette question."},
    {"role": "user", "content": m2},
    {"role": "assistant", "content": r2}
]

# CORRECTION 1 : On crée une copie propre de la liste pour éviter une liste dans une liste
history = list(config)
seuil = 20

def model(message):
    global history
    history.append({"role": "user", "content": message})

    # CORRECTION 2 : Ta logique de nettoyage (on garde les 5 messages de config de base)
    while len(history) - 3 > seuil:
        history.pop(3)  # On retire le message juste après la configuration de base
 
    # CORRECTION 3 : Alignement parfait de l'indentation sous la fonction
    try:
        response = client.chat.completions.create(
            model='dots-studio/dots-3-note-preview:free', 
            messages=history,
            timeout=30,
            extra_body={'reasoning': {"exclude": True}} # CORRECTION 4 : 'exclude' correctement orthographié
        )
        reponse_ia = response.choices[0].message.content
        history.append({"role": "assistant", "content": reponse_ia})
        return reponse_ia

    # CORRECTION 5 : Les erreurs spécifiques sont placées TOUT EN HAUT du filet de sécurité
    except openai.APITimeoutError:
        return 'Le modèle prend trop de temps pour répondre, veuillez réessayer plus tard.'

    except openai.APIConnectionError:
        return 'Problème de connexion avec le serveur, veuillez réessayer plus tard.'

    except openai.AuthenticationError:
        return 'Erreur d\'authentification (clé API invalide), veuillez réessayer plus tard.'
        
    except Exception as e:
        # L'erreur générale intercepte tout le reste en dernier recours
        return f"Désolé on rencontre des soucis actuellement, veuillez réessayer plus tard. (Détail: {e})"

# --- REMPLACE LA DERNIÈRE LIGNE PAR CE BLOC ---
if __name__ == "__main__":
    print("🦁 Chatbot Zoologue Activé ! (Tape 'quitter' pour arrêter) \n")
    
    while True:
        # Le programme s'arrête ici et attend que tu tapes ton texte au clavier
        ma_question = input("vous avez une question?  : ")
        
        # Si tu tapes 'quitter', on sort de la boucle et le programme s'arrête
        if ma_question.lower() == "quitter":
            print("Au revoir !")
            break
            
        # On envoie la question au modèle et on récupère sa réponse
        reponse_de_l_ia = model(ma_question)
        
        # On affiche la réponse
        print(f"Agent : {reponse_de_l_ia}\n")
        print("-" * 50) # Une petite ligne de séparation pour que ce soit joli
