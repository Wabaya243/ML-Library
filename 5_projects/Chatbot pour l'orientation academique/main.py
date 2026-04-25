r"""
Serveur FastAPI minimal pour dialoguer avec l'app Flutter et Api_test.py.

- POST /chat        : réponse non-streaming (JSON)
- POST /chat/stream : réponse en streaming (SSE)

Le serveur attend un payload de la forme:
{
  "message": "dernier message utilisateur",
  "messages": [ {"role":"system|user|assistant","content":"..."}, ... ],
  "stream": true|false
}

Notes:
- CORS activé pour faciliter les tests (localhost, 127.0.0.1, 10.0.2.2, etc.).
- La "mémoire" est gérée côté client: envoyez simplement les 5 derniers messages.
- Remplacez la fonction `generate_reply` par un appel à votre modèle.


on lance par 

cd  C:\Users\OMEN\Documents\Cours I.A (Machine Learning)\5. Projet\Chatbot pour l'orientation academique
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
from typing import Dict, List, Any
import logging
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from model_service import ModelService
import traceback


# Création de l’application FastAPI principale
app = FastAPI(title="Academia Chat API", version="0.1.0")

# Activation de CORS pour permettre à Flutter / navigateur local d’appeler l’API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # En dev, on autorise tout. En prod, à restreindre aux domaines fiables
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
        "http://10.0.2.2",
        "http://10.0.2.2:8000",
    ],
    allow_credentials=True,  # Autorise les cookies / tokens
    allow_methods=["*"],     # Toutes les méthodes HTTP (POST, GET, etc.)
    allow_headers=["*"],     # Tous les headers (utile pour JSON, Authorization, etc.)
)


def _last_k_ua(messages: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    r"""Filtre et conserve uniquement les K derniers échanges user/assistant."""
    # On ne garde pas les messages "system" ici car ils servent juste d’instruction globale
    ua = [m for m in messages if m.get("role") in ("user", "assistant")]
    # Retourne les K derniers échanges pour limiter le contexte (évite surcharge mémoire)
    return ua[-k:]


def build_context(payload: Dict[str, Any], memory: int = 5) -> List[Dict[str, str]]:
    r"""Construit le contexte complet pour le modèle: [system] + derniers K échanges + question courante."""

    # Récupère les messages précédents envoyés par le client (historique côté Flutter)
    history = payload.get("messages") or []

    # Récupère le dernier message utilisateur actuel
    message = payload.get("message") or ""

    # Vérifie si un message "system" existe déjà au début de la conversation
    if history and isinstance(history[0], dict) and history[0].get("role") == "system":
        system = history[0]  # Réutilise celui fourni par le client
    else:
        # Sinon, injecte une consigne standard (garantit un comportement constant du modèle)
        system = {
            "role": "system",
            "content": ("Tu es un conseiller d’orientation académique expérimenté. "
                    "Réponds toujours à la dernière question de l'utilisateur "
                    "en tenant compte du contexte précédent, sans inventer, Réponds de façon claire, contextualisée et responsable et explicatif")
        }

    # Garde les derniers échanges user/assistant pour le contexte
    # Conserve les 6 dernières paires (user+assistant) = 12 messages max
    memory_pairs = 8
    tail = _last_k_ua(history[1:] if len(history) > 1 else [], memory_pairs * 2)

    # Ajoute le nouveau message utilisateur si pas encore inclus dans l’historique
    if not tail or tail[-1].get("role") != "user" or tail[-1].get("content") != message:
        tail.append({"role": "user", "content": message})

    # Retourne le contexte complet dans l’ordre: system → échanges récents → question actuelle
    return [system] + tail


def generate_reply(context: List[Dict[str, str]]) -> str:
    """Produit une réponse à partir du modèle local, ou une phrase de secours si erreur."""
    try:
        # Appelle l’instance du modèle (chargée via ModelService)
        svc = ModelService.instance()
        return svc.generate(context)
    except Exception as e:
        # En cas d’erreur (modèle non chargé, crash, etc.), on renvoie un message de secours
        user_last = next((m["content"] for m in reversed(context) if m.get("role") == "user"), "")
        return f"[fallback] Réponse de démonstration à: \"{user_last}\" (err: {e})"


@app.post("/chat")
async def chat(payload: Dict[str, Any]):
    """Endpoint non-streaming: retourne la réponse complète sous forme JSON."""
    # Construit le contexte conversationnel pour le modèle
    context = build_context(payload, memory=5)
    # Génère la réponse à partir du modèle local ou fallback
    reply = generate_reply(context)
    # Envoie la réponse en JSON complet (utile pour les tests rapides ou API simples)
    return JSONResponse({"reply": reply})


@app.post("/chat/stream")
async def chat_stream(payload: Dict[str, Any]):
    """Endpoint streaming SSE: envoie la réponse du modèle token par token (type ChatGPT)."""

    # Prépare le contexte conversationnel (mêmes règles que ci-dessus)
    context = build_context(payload, memory=5)

    # Fallback d’urgence: en cas d’erreur modèle, on renvoie la réponse token par token manuellement
    async def _fallback_echo():
        full = generate_reply(context)  # Génère le texte complet
        await asyncio.sleep(0.15)       # Pause initiale pour simuler la latence
        for token in full.split():
            data = json.dumps({"delta": {"content": token + " "}})
            yield f"data: {data}\n\n"   # Format SSE: chaque token envoyé séparément
            await asyncio.sleep(0.02)   # Petite pause entre chaque token
        yield "data: [DONE]\n\n"        # Indique la fin du flux

    try:
        # On tente d’utiliser le modèle avec génération en streaming
        svc = ModelService.instance()
        stream_iter = svc.stream(context)

        async def event_gen_model():
            try:
                for chunk in stream_iter:
                    data = json.dumps({"delta": {"content": chunk}})
                    yield f"data: {data}\n\n"  # Envoie chaque fragment SSE au client
                    await asyncio.sleep(0)     # Laisse tourner l’event loop
                yield "data: [DONE]\n\n"       # Fin propre du flux
            except Exception:
                # Si le modèle plante en cours de route, on affiche l’erreur et bascule sur le fallback
                traceback.print_exc()
                async for x in _fallback_echo():
                    yield x

        # Envoie la réponse sous forme d’un flux SSE continu
        return StreamingResponse(event_gen_model(), media_type="text/event-stream")
    except Exception:
        # Si tout échoue (ex: modèle non dispo), on log et lance le fallback
        traceback.print_exc()
        return StreamingResponse(_fallback_echo(), media_type="text/event-stream")


@app.get("/health")
async def health():
    """Simple endpoint pour vérifier que le serveur tourne."""
    return {"status": "ok"}


@app.get("/debug/info")
async def debug_info():
    """Permet de récupérer des infos sur le modèle courant (pour debug)."""
    try:
        svc = ModelService.instance()
        return svc.info()
    except Exception as e:
        return {"error": str(e)}


# -------------------------
# Commentaires explicites
# -------------------------
# - /chat        : endpoint non‑streaming. Retourne {"reply": "..."} en une fois.
# - /chat/stream : endpoint streaming SSE. Envoie des lignes 'data: {json}' puis 'data: [DONE]'.
# - build_context: construit le contexte [system] + derniers messages + question courante.
#   Si aucun message system n'est fourni par le client, on injecte un message system
#   aligné avec le script d'entraînement (test_chatbot) pour conserver le rôle conseillé.
# - Suppression de <END> : côté non‑stream (ModelService.generate) et côté stream (remplacement
#   dans la boucle) on filtre ce token spécial pour ne pas l'afficher dans l'UI.
# - CORS: activé en dev pour permettre les appels depuis 127.0.0.1, 10.0.2.2, etc.
# - Erreurs serveur: en cas d'exception dans le streaming, on journalise la stacktrace
#   et on renvoie un flux de secours (fallback_echo) pour éviter de couper la connexion.
