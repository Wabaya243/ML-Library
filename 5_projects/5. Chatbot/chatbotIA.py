import nltk
import re
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline

# Téléchargement des ressources NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Configuration du device (GPU si dispo)
device = 0 if torch.cuda.is_available() else -1

# Chargement du modèle français
chatbot = pipeline("text-generation", model="cedpsam/chatbot_fr", device=device)

# Nettoyage du texte
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("french")]
    cleaned_text = " ".join(tokens)
    return cleaned_text


# Chatbot basé sur des règles simples
def simple_chatbot(user_input):
    user_input = clean_text(user_input)
    responses = {
        "bonjour": "Bonjour ! Comment puis-je vous aider ?",
        "comment vas tu": "Je vais bien, merci !",
        "au revoir": "Au revoir ! N'hésitez pas à revenir si vous avez besoin d'aide.",
        "merci": "De rien ! N'hésitez pas à me poser d'autres questions."
    }
    for key in responses:
        if key in user_input:
            return responses[key]
    return "Je suis désolé, je ne comprends pas votre question."

# Exemple
print(simple_chatbot("bonjour"))

# Chatbot basé sur le modèle Hugging Face
def ai_chatbot(user_input):
    response = chatbot(
        user_input,
        max_new_tokens=30,   # <--- génère moins de texte
        temperature=0.7,     # <--- rend la réponse plus cohérente
        top_p=0.9,           # <--- échantillonnage plus concentré
        num_return_sequences=1
    )
    return response[0]["generated_text"]


# Boucle principale
def chatbot_system():
    print("Chatbot : Bonjour ! En quoi puis-je vous aider ? (tapez 'stop' pour quitter)")
    while True:
        user_input = input("Vous : ")
        if user_input.lower() == "stop":
            print("Chatbot : Au revoir !")
            break
        elif any(word in user_input.lower() for word in ["bonjour", "comment vas tu", "au revoir", "merci"]):
            print("Chatbot :", (user_input))
        else:
            response = ai_chatbot(user_input)
            print("Chatbot :", response)


chatbot_system()









