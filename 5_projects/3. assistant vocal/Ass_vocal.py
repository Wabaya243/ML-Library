import speech_recognition as sr
import pyttsx3
from datetime import datetime
import wikipedia
import time


engine = pyttsx3.init(driverName='sapi5')   # Windows

def speak(text):
    engine = pyttsx3.init()  # moteur réinitialisé à chaque fois
    voices = engine.getProperty('voices')
    for v in voices:
        if 'fr' in v.id.lower():
            engine.setProperty('voice', v.id)
            break
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def get_time():
    t = datetime.now().strftime("%H:%M:%S")
    speak(f"Il est actuellement {t}")
    print(f"Il est actuellement {t}")

def search_wikipedia(query):
    wikipedia.set_lang("fr")
    try:
        result = wikipedia.summary(query, sentences=2)
        speak(result)
        print(result)
    except wikipedia.exceptions.DisambiguationError as e:
        speak("Il y a plusieurs articles possibles. Pouvez-vous être plus précis ?")
        print(e.options)
    except wikipedia.exceptions.PageError:
        speak("Désolé, je n'ai pas trouvé d'article sur ce sujet.")
        print("Désolé, je n'ai pas trouvé d'article sur ce sujet.")

def process_command(command):
    if "heure" in command:
        time.sleep(0.5)
        get_time()
    elif "wikipedia" in command or "wikipédia" in command:
        speak("Quel sujet souhaitez-vous savoir sur Wikipédia ?")
        query = recognize_speech()
        if query:
            search_wikipedia(query)
    elif any(word in command for word in ["arret", "arrête", "stop", "quitte", "ferme"]):
        speak("Merci d'avoir utilisé l'assistant vocal. À bientôt !")
        time.sleep(1)
        raise SystemExit  # ← quitte proprement la boucle
    else:
        speak("Cette commande n'est pas reconnue.")

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("En écoute...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language="fr-FR")
        print("Vous avez dit :", text)
        return text.lower()
    except sr.UnknownValueError:
        print("Désolé, je n'ai pas compris.")
        return None
    except sr.RequestError as e:
        print(f"Erreur de service : {e}")
        return None

def start_voice():
    speak("Bonjour, je suis votre assistant vocal. Comment puis-je vous aider ?")
    time.sleep(1.5)
    while True:
        speak("J'écoute.")
        command = recognize_speech()
        if command:
            process_command(command)

start_voice()
