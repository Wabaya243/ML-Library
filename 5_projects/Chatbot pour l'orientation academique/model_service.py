"""
Service de modèle pour Mistral 7B Instruct + LoRA, avec génération
non-streaming et streaming via TextIteratorStreamer.

Par défaut, on charge le modèle de base "mistralai/Mistral-7B-Instruct-v0.3"
et on applique l'adapter LoRA local (adapter_path) — par défaut le dossier
"mistral_unikin_lora2" situé au même niveau que main.py.

Prérequis conda/pip:
  pip install torch transformers peft bitsandbytes
  (CUDA recommandé pour des performances correctes)

Variables d'environnement (optionnelles):
  MODEL_ADAPTER_PATH : chemin absolu vers le dossier LoRA
  MODEL_LOAD_4BIT    : "1"/"true" pour activer 4-bit (défaut: 1), sinon "0"
"""

from __future__ import annotations  # permet d'annoter des types avec des classes définies plus bas

import logging
import os
import threading
from typing import Dict, List, Iterable, Optional

import torch
from transformers import (
    AutoModelForCausalLM,     # charge un modèle CausalLM depuis HF Hub/local
    AutoTokenizer,            # charge le tokenizer associé
    BitsAndBytesConfig,       # config de quantification 4-bit (bitsandbytes)
    TextIteratorStreamer,     # utilitaire pour streamer la génération token-par-token
)
from peft import PeftModel      # applique un adapter LoRA à un modèle de base
from typing import Any

log = logging.getLogger(__name__)  # logger module-scoped (pratique pour debug)

class ModelService:
    _instance: Optional["ModelService"] = None  # singleton (une seule instance vivante)

    def __init__(
        self,
        base_model: str = "mistralai/Mistral-7B-Instruct-v0.3",  # nom du modèle de base
        adapter_path: Optional[str] = None,                      # chemin vers l'adapter LoRA
        load_4bit: Optional[bool] = None,                        # activer ou non la quantification 4-bit
    ) -> None:
        # Détermine le dossier projet (emplacement de ce fichier)
        proj_dir = os.path.abspath(os.path.dirname(__file__))
        # Adapter par défaut : dossier "mistral-unikin-lora2" à côté du code
        default_adapter = os.path.join(proj_dir, "mistral-unikin-lora")
        # Variables d’environnement optionnelles (permet de surcharger sans modifier le code)
        env_adapter = os.getenv("MODEL_ADAPTER_PATH")
        env_load4 = os.getenv("MODEL_LOAD_4BIT")

        # Stocke les paramètres choisis / résolus
        self.base_model = base_model
        # Priorité : arg → env → défaut
        self.adapter_path = adapter_path or env_adapter or default_adapter
        # 4-bit actif par défaut si MODEL_LOAD_4BIT est absent ou "true"/"1"
        self.load_4bit = (load_4bit if load_4bit is not None else (env_load4 is None or env_load4.lower() in {"1","true","yes"}))
        self.tok = None  # sera peuplé par _load()
        self.mdl = None  # idem

    @classmethod
    def instance(cls) -> "ModelService":
        """Retourne l’instance unique (singleton), charge le modèle au premier appel."""
        if cls._instance is None:
            cls._instance = ModelService()  # construit avec les valeurs par défaut/env
            cls._instance._load()           # charge modèle + tokenizer + LoRA
        return cls._instance

    # Chargement du modèle/tokenizer une seule fois
    def _load(self) -> None:
        """Charge tokenizer, modèle de base (option 4-bit) et applique l'adapter LoRA."""
        if not os.path.isdir(self.adapter_path):
            # Adapter introuvable : on lève une erreur descriptive
            raise FileNotFoundError(f"Adapter LoRA introuvable: {self.adapter_path}")

        # Vérifie si le dossier adapter contient un tokenizer (tokenizer.json, vocab, etc.)
        adapter_has_tokenizer = False
        try:
            if os.path.isdir(self.adapter_path):
                entries = set(os.listdir(self.adapter_path))
                adapter_has_tokenizer = any(name.startswith("tokenizer") for name in entries) or "tokenizer.json" in entries
        except Exception:
            adapter_has_tokenizer = False

        # Charge le tokenizer depuis l’adapter (si présent) sinon depuis le modèle de base
        if adapter_has_tokenizer:
            log.info("Chargement du tokenizer depuis l'adapter: %s", self.adapter_path)
            self.tok = AutoTokenizer.from_pretrained(self.adapter_path, use_fast=True)
        else:
            log.info("Chargement du tokenizer depuis le modèle de base: %s", self.base_model)
            self.tok = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)

        # Garantit l’existence d’un pad_token (évite des warnings/erreurs à l’inférence)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        # Prépare la config de quantification si 4-bit activé
        quant_config = None
        if self.load_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,                    # active la quantification 4-bit
                bnb_4bit_compute_dtype=torch.bfloat16,# calcul en bfloat16 (rapide/précis sur GPU récents)
                bnb_4bit_use_double_quant=True,       # double quantification (réduit encore VRAM)
                bnb_4bit_quant_type="nf4",            # schéma NF4 (qualité souvent supérieure à FP4)
            )

        log.info("Chargement du modèle de base %s (4-bit=%s)", self.base_model, self.load_4bit)
        self.mdl = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quant_config,  # None si 4-bit off → chargement full
            device_map="auto",                 # place automatiquement sur GPU/CPU disponibles
        )

        # IMPORTANT: si le tokenizer (adapter) ajoute des tokens (ex: "<END>"),
        # on ajuste la taille des embeddings du modèle pour éviter mismatch.
        self.mdl.resize_token_embeddings(len(self.tok))

        log.info("Application de l'adapter LoRA %s", self.adapter_path)
        # Applique les poids LoRA sur le modèle de base
        self.mdl = PeftModel.from_pretrained(self.mdl, self.adapter_path)
        # L’inférence ne requiert pas de gradient
        self.mdl.eval()

        # Logs d’auto-check LoRA (liste des adapters et celui actif)
        try:
            peft_keys = list(getattr(self.mdl, "peft_config", {}).keys())
        except Exception:
            peft_keys = []
        active = None
        try:
            active = getattr(self.mdl, "active_adapters", None)
        except Exception:
            active = None

        log.info("PEFT adapters: %s | actif: %s", peft_keys, active)
        log.info("Modèle chargé et prêt.")

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Construit le prompt selon le chat template si dispo, sinon fallback simple."""
        try:
            # Utilise le template fourni par le tokenizer (généralement > qualité)
            return self.tok.apply_chat_template(
                messages,
                add_generation_prompt=True,  # ajoute le tour assistant attendu
                tokenize=False,              # retourne un string, pas des ids
            )
        except Exception:
            # Fallback très simple si pas de template : concatène rôles/contents
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"<{role}>: {content}")
            return "\n".join(parts) + "\n<assistant>:"

    def generate(self, messages: List[Dict[str, str]], **gen_kwargs) -> str:
        """Génération non-streaming : renvoie tout le texte d’un coup (nettoie <END>)."""
        prompt = self._build_prompt(messages)                 # construit prompt
        inputs = self.tok(prompt, return_tensors="pt").to(self.mdl.device)  # encode et envoie au device
        eos_ids = [self.tok.eos_token_id]                    # EOS par défaut
        # Essaie d’ajouter l’ID du token spécial <END> si présent dans le vocab
        try:
            end_id = self.tok.convert_tokens_to_ids("<END>")
            if isinstance(end_id, int) and end_id > 0:
                eos_ids.append(end_id)
        except Exception:
            pass

        # Hyperparamètres par défaut raisonnables (peuvent être surchargés via gen_kwargs)
        defaults = dict(
            max_new_tokens=250,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=eos_ids,  # stop soit sur </s>, soit sur <END>
        )
        defaults.update(gen_kwargs)

        # Génération en no_grad (pas de backprop à l’inférence)
        with torch.no_grad():
            out = self.mdl.generate(**inputs, **defaults)

        # On ne décode que les nouveaux tokens (hors prompt)
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        text = self.tok.decode(new_tokens, skip_special_tokens=True)

        # Coupe proprement si <END> apparaît dans la sortie
        text = text.split("<END>")[0].strip()
        return text

    def stream(self, messages: List[Dict[str, str]], **gen_kwargs) -> Iterable[str]:
        """Génération streaming : yield des segments de texte (pour SSE côté API)."""
        prompt = self._build_prompt(messages)
        inputs = self.tok(prompt, return_tensors="pt").to(self.mdl.device)
        eos_ids = [self.tok.eos_token_id]
        try:
            end_id = self.tok.convert_tokens_to_ids("<END>")
            if isinstance(end_id, int) and end_id > 0:
                eos_ids.append(end_id)
        except Exception:
            pass

        # Streamer HF : gère le buffering et renvoie des morceaux de texte au fil de l’eau
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)

        # Paramètres par défaut (identiques au non-streaming, mais avec streamer)
        defaults = dict(
            max_new_tokens=250,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=eos_ids,
            streamer=streamer,  # c'est le streamer qui capture la sortie
        )
        defaults.update(gen_kwargs)

        # Le generate() bloque tant qu’il produit → on le lance dans un thread dédié
        def _worker():
            with torch.no_grad():
                self.mdl.generate(**inputs, **defaults)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()  # démarre la génération en arrière-plan

        # Boucle consommateur : on itère sur le streamer et on yield chaque fragment
        for text in streamer:
            yield text

    def info(self) -> Dict[str, Any]:
        """Renvoie des infos debug sur le chargement du modèle/adapter."""
        data: Dict[str, Any] = {
            "base_model": self.base_model,                   # nom du modèle de base
            "adapter_path": self.adapter_path,               # chemin LoRA
            "load_4bit": self.load_4bit,                     # 4-bit on/off
            "tokenizer_len": len(self.tok) if self.tok is not None else None,  # taille vocab
        }
        # Liste des adapters LoRA connus par PEFT
        try:
            data["peft_adapters"] = list(getattr(self.mdl, "peft_config", {}).keys())
        except Exception:
            data["peft_adapters"] = None
        # Adapter(s) actif(s) si dispo
        try:
            data["active_adapters"] = getattr(self.mdl, "active_adapters", None)
        except Exception:
            data["active_adapters"] = None
        # ID du token <END> (utile pour vérifier la cohérence tokenizer)
        try:
            end_id = self.tok.convert_tokens_to_ids("<END>") if self.tok is not None else None
        except Exception:
            end_id = None
        data["end_token_id"] = end_id
        return data

# -------------------------
# Commentaires explicites
# -------------------------
# Étapes de chargement (_load):
# 1) Tokenizer: depuis l'adapter si présent, sinon depuis le modèle de base.
#    - On définit pad_token si absent.
# 2) Modèle: chargement du modèle de base (option 4-bit selon MODEL_LOAD_4BIT),
#    puis redimension des embeddings selon la taille du tokenizer (évite mismatch).
# 3) LoRA: application de l'adapter via PEFT (PeftModel.from_pretrained), puis eval().

# Inférence:
# - generate(): construit le prompt (chat template si dispo), génère en bloc et coupe à <END>.
# - stream(): même prompt/params mais renvoie des segments via TextIteratorStreamer (pour SSE).

# Dépannage:
# - FileNotFoundError adapter → vérifier MODEL_ADAPTER_PATH ou le dossier 'mistral-unikin-lora2'.
# - Mismatch embedding (size mismatch) → s’assurer d’appeler resize_token_embeddings après le tokenizer.
# - bitsandbytes/Windows → si l’init 4-bit échoue, mettre MODEL_LOAD_4BIT=0 (ou load_4bit=False).
# - Sortie “bavarde”/hors sujet → vérifier le message system et la cohérence du chat template.
