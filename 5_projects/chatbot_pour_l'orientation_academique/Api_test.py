"""
Serveur FastAPI minimal pour dialoguer avec l'app Flutter et Api_test.py.

- POST /chat        : réponse non-streaming (JSON complet)
- POST /chat/stream : réponse en streaming (SSE, style ChatGPT)

Le serveur attend un payload de la forme:
{
  "message": "dernier message utilisateur",
  "messages": [ {"role":"system|user|assistant","content":"..."}, ... ],
  "stream": true|false
}

Notes:
- CORS activé pour faciliter les tests (localhost, 127.0.0.1, 10.0.2.2, etc.)
- La mémoire du chat est gérée côté client: on n’envoie que les derniers messages pertinents.
- `generate_reply` utilise le modèle DeepSeek-R1-Distill-Qwen-7B + LoRA fine-tuné (UNIKIN Advisor).
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
from model_service import ModelService  # version adaptée DeepSeek
import traceback

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ----------------------------------------
# INITIALISATION DE L’APPLICATION
# ----------------------------------------

# Création de l’application FastAPI (titre et version affichés dans la doc /docs)
app = FastAPI(title="DeepSeek UNIKIN Chat API", version="1.0.0")

# Configuration CORS : permet à ton app Flutter et ton navigateur de contacter le backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # En dev on autorise tout (à restreindre plus tard pour la sécurité)
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
        "http://10.0.2.2",        # Adresse spéciale de l’émulateur Android vers l’hôte
        "http://10.0.2.2:8000",
    ],
    allow_credentials=True,  # Autorise l’envoi de cookies ou tokens JWT
    allow_methods=["*"],     # Autorise toutes les méthodes HTTP
    allow_headers=["*"],     # Autorise tous les headers (utile pour JSON)
)


# ----------------------------------------
# GESTION DU CONTEXTE DE CONVERSATION
# ----------------------------------------

def _last_k_ua(messages: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Filtre et conserve uniquement les K derniers échanges user/assistant."""
    # On ne garde pas les messages "system" (ils ne sont pas partie du dialogue utilisateur)
    ua = [m for m in messages if m.get("role") in ("user", "assistant")]
    # On limite le nombre de tours pour ne pas surcharger la mémoire du modèle
    return ua[-k:]


def build_context(payload: Dict[str, Any], memory: int = 5) -> List[Dict[str, str]]:
    """Construit le contexte complet envoyé au modèle (system + derniers échanges + message actuel)."""

    # Historique complet fourni par le client (vient du frontend Flutter)
    history = payload.get("messages") or []

    # Dernier message saisi par l’utilisateur
    message = payload.get("message") or ""

    # Si le tout premier message est un "system", on le garde
    if history and isinstance(history[0], dict) and history[0].get("role") == "system":
        system = history[0]
    else:
        # Sinon, on insère un message système standard pour cadrer le ton du modèle
        system = {
            "role": "system",
            "content": (
                "Tu es un conseiller académique de l’Université de Kinshasa. "
                 "Réponds dans un français fluide, logique et bienveillant. "
                 "Sois clair, rigoureux et explicatif. N’utilise jamais l’anglais, "
                "même si la question est en anglais : traduis toujours et réponds en français uniquement."
            )
        }

    # On garde uniquement les derniers échanges entre user et assistant
    tail = _last_k_ua(history[1:] if len(history) > 1 else [], memory)

    # Si le dernier message utilisateur n’est pas déjà inclus, on l’ajoute
    if not tail or tail[-1].get("role") != "user" or tail[-1].get("content") != message:
        tail.append({"role": "user", "content": message})

    # On retourne la liste complète dans l’ordre logique pour le modèle
    return [system] + tail


# ----------------------------------------
# GÉNÉRATION DE RÉPONSE
# ----------------------------------------

def generate_reply(context: List[Dict[str, str]]) -> str:
    """Appelle le modèle DeepSeek LoRA pour générer une réponse ou renvoie un fallback si erreur."""
    try:
        svc = ModelService.instance()  # Singleton vers le modèle déjà chargé en mémoire
        return svc.generate(context)   # Génération directe (non-streaming)
    except Exception as e:
        # Si le modèle n’est pas disponible, on renvoie une phrase de secours
        user_last = next((m["content"] for m in reversed(context) if m.get("role") == "user"), "")
        return f"[fallback] Réponse simulée à: \"{user_last}\" (erreur: {e})"


# ----------------------------------------
# ENDPOINT NON-STREAMING
# ----------------------------------------

@app.post("/chat")
async def chat(payload: Dict[str, Any]):
    """Endpoint simple : retourne la réponse complète d’un coup (format JSON)."""
    context = build_context(payload, memory=5)  # Construit le contexte propre
    reply = generate_reply(context)             # Appelle le modèle
    return JSONResponse({"reply": reply})       # Réponse directe (pour test simple, API_test.py, etc.)


# ----------------------------------------
# ENDPOINT STREAMING SSE
# ----------------------------------------

@app.post("/chat/stream")
async def chat_stream(payload: Dict[str, Any]):
    """Endpoint SSE : envoie la réponse token par token pour un affichage progressif."""
    context = build_context(payload, memory=5)

    # Fallback en cas d'erreur du modèle : simule un flux token par token
    async def _fallback_echo():
        full = generate_reply(context)  # Génère la réponse complète
        await asyncio.sleep(0.15)       # Petit délai pour simuler la latence
        for token in full.split():
            data = json.dumps({"delta": {"content": token + " "}})
            yield f"data: {data}\n\n"   # Chaque token envoyé sous forme SSE
            await asyncio.sleep(0.02)   # Simule la vitesse d’écriture humaine
        yield "data: [DONE]\n\n"        # Signal de fin du flux

    try:
        # Tentative d’appel au modèle réel avec streaming
        svc = ModelService.instance()
        stream_iter = svc.stream(context)  # Itérateur renvoyant les morceaux générés

        async def event_gen_model():
            try:
                for chunk in stream_iter:
                    data = json.dumps({"delta": {"content": chunk}})
                    yield f"data: {data}\n\n"  # Envoie chaque portion SSE
                    await asyncio.sleep(0)     # Laisse tourner l’event loop
                yield "data: [DONE]\n\n"       # Fin du flux proprement
            except Exception:
                # Si le modèle plante en cours de flux, on log et bascule sur le fallback
                traceback.print_exc()
                async for x in _fallback_echo():
                    yield x

        # Retourne le flux SSE au client Flutter / Web
        return StreamingResponse(event_gen_model(), media_type="text/event-stream")
    except Exception:
        # En cas d’erreur fatale (modèle introuvable ou crash), fallback total
        traceback.print_exc()
        return StreamingResponse(_fallback_echo(), media_type="text/event-stream")


# ----------------------------------------
# ENDPOINTS DE MAINTENANCE
# ----------------------------------------

@app.get("/health")
async def health():
    """Vérifie l’état de santé du serveur (utile pour Docker ou tests automatiques)."""
    # Si on arrive ici sans erreur, c’est que l’API tourne
    return {"status": "ok"}  # Simple ping de disponibilité


@app.get("/debug/info")
async def debug_info():
    """Expose des informations sur le modèle actuellement chargé (debug / monitoring)."""
    try:
        svc = ModelService.instance()  # Récupère l’instance unique du modèle
        return svc.info()              # Retourne ses infos (nom, taille, GPU utilisé, etc.)
    except Exception as e:
        # En cas d’erreur, renvoie un message explicite pour ne pas planter le serveur
        return {"error": str(e)}
