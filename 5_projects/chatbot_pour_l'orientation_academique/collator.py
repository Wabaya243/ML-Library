

import os, torch, warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*h5py is running against HDF5.*")



from dataclasses import dataclass
from transformers import DataCollatorWithPadding


#Deepseek
@dataclass
class DataCollatorAssistantOnly:
    tokenizer: object
    assistant_header: str = "<｜Assistant｜>"
    end_token: str = "<|EOT|>"

    def __call__(self, features):
        pad = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")
        batch = pad(features)
        input_ids = batch["input_ids"]
        labels = torch.full_like(input_ids, -100)

        start_pat = self.tokenizer.encode(self.assistant_header, add_special_tokens=False)
        end_pat = self.tokenizer.encode(self.end_token, add_special_tokens=False)

        for i, ids in enumerate(input_ids):
            ids_list = ids.tolist()
            start = -1
            for j in range(len(ids_list) - len(start_pat) + 1):
                if ids_list[j:j + len(start_pat)] == start_pat:
                    start = j + len(start_pat)
                    break
            if start == -1:
                continue
            end = len(ids_list)
            for k in range(start, len(ids_list) - len(end_pat) + 1):
                if ids_list[k:k + len(end_pat)] == end_pat:
                    end = k
                    break
            labels[i, start:end] = input_ids[i, start:end]

        batch["labels"] = labels
        return batch



#Gwen
@dataclass
class DataCollatorAssistantOnlyGwen:
    tokenizer: object
    assistant_header: str = "<|im_start|>assistant"
    end_token: str = "<|im_end|>"

    def __call__(self, features):
        pad = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")
        batch = pad(features)
        input_ids = batch["input_ids"]
        labels = torch.full_like(input_ids, -100)

        start_pat = self.tokenizer.encode(self.assistant_header, add_special_tokens=False)
        end_pat   = self.tokenizer.encode(self.end_token,      add_special_tokens=False)

        for i, ids in enumerate(input_ids.tolist()):
            # cherche le dernier <|im_start|>assistant
            start = -1
            for j in range(len(ids) - len(start_pat) + 1):
                if ids[j:j+len(start_pat)] == start_pat:
                    start = j + len(start_pat)
            if start == -1:
                continue

            # cherche <|im_end|> après start
            end = len(ids)
            for k in range(start, len(ids) - len(end_pat) + 1):
                if ids[k:k+len(end_pat)] == end_pat:
                    end = k
                    break

            labels[i, start:end] = torch.tensor(input_ids[i, start:end])
        batch["labels"] = labels
        return batch


## Deepthink
@dataclass
class DataCollatorAssistantOnlyDeepthink:
    tokenizer: object
    assistant_header: str = "<|im_start|>assistant"
    end_token: str = "<|im_end|>"

    def __call__(self, features):
        pad = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")
        batch = pad(features)
        input_ids = batch["input_ids"]
        labels = torch.full_like(input_ids, -100)

        start_pat = self.tokenizer.encode(self.assistant_header, add_special_tokens=False)
        end_pat = self.tokenizer.encode(self.end_token, add_special_tokens=False)

        for i, ids in enumerate(input_ids.tolist()):
            start = -1
            for j in range(len(ids) - len(start_pat) + 1):
                if ids[j:j+len(start_pat)] == start_pat:
                    start = j + len(start_pat)
            if start == -1:
                continue
            end = len(ids)
            for k in range(start, len(ids) - len(end_pat) + 1):
                if ids[k:k+len(end_pat)] == end_pat:
                    end = k
                    break
            labels[i, start:end] = torch.tensor(input_ids[i, start:end])
        batch["labels"] = labels
        return batch


## Claire

@dataclass
class DataCollatorAssistantOnlyClaire:
    tokenizer: object
    assistant_header: str = "[/INST]"
    end_token: str = "</s>"

    def __call__(self, features):
        pad = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")
        batch = pad(features)
        input_ids = batch["input_ids"]
        labels = torch.full_like(input_ids, -100)

        start_pat = self.tokenizer.encode(self.assistant_header, add_special_tokens=False)
        end_pat = self.tokenizer.encode(self.end_token, add_special_tokens=False)

        for i, ids in enumerate(input_ids.tolist()):
            start = -1
            for j in range(len(ids) - len(start_pat) + 1):
                if ids[j:j+len(start_pat)] == start_pat:
                    start = j + len(start_pat)
            if start == -1:
                continue
            end = len(ids)
            for k in range(start, len(ids) - len(end_pat) + 1):
                if ids[k:k+len(end_pat)] == end_pat:
                    end = k
                    break
            labels[i, start:end] = torch.tensor(input_ids[i, start:end])
        batch["labels"] = labels
        return batch
    
    
    
## YI 


# ----------------------------------------------------------
# Collator — n'entraîner QUE la zone assistant (ChatML Yi)
# Yi-1.5-Chat utilise un template ChatML: <|im_start|>role ... <|im_end|>
# ----------------------------------------------------------
@dataclass
class DataCollatorAssistantOnlyYI:
    tokenizer: object
    end_token: str = "<|im_end|>"
    assistant_header: str = "<|im_start|>assistant"

    def __call__(self, features):
        pad = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")
        batch = pad(features)
        input_ids = batch["input_ids"]
        labels = torch.full_like(input_ids, -100)

        end_id = self.tokenizer.convert_tokens_to_ids(self.end_token)
        pat = self.tokenizer.encode(self.assistant_header, add_special_tokens=False)

        for i, ids in enumerate(input_ids):
            ids_list = ids.tolist()

            # position du DERNIER début de tour assistant
            inst_pos = -1
            for j in range(len(ids_list) - len(pat) + 1):
                if ids_list[j:j+len(pat)] == pat:
                    inst_pos = j + len(pat)
            if inst_pos == -1:
                continue

            # premier <|im_end|> après inst_pos
            try:
                end_pos = ids_list.index(end_id, inst_pos + 1)
            except ValueError:
                end_pos = len(ids_list)

            if end_pos > inst_pos:
                labels[i, inst_pos:end_pos] = input_ids[i, inst_pos:end_pos]

        batch["labels"] = labels
        return batch

# DataCollatorAssistantOnly — pour LLaMA 3.1 / DeepSeek / Qwen

from dataclasses import dataclass
from typing import Dict, List
import torch
from transformers import PreTrainedTokenizerBase

@dataclass
class DataCollatorAssistantOnlyLLamma:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        # Convertir la liste de dicts {"text": "..."} → encodage batch
        texts = [f["text"] for f in features]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=540,
            return_tensors="pt"
        )

        # On masque tout sauf la partie "assistant"
        # Les modèles de type ChatML encodent souvent l’assistant après le dernier token "assistant"
        labels = batch["input_ids"].clone()

        # ID du token de rôle <|assistant|> ou similaire
        role_ids = []
        for i, ids in enumerate(batch["input_ids"]):
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            start_idx = 0
            for j, tok in enumerate(tokens):
                if "assistant" in tok or "<|im_start|>assistant" in tok:
                    start_idx = j
                    break
            # On masque tout avant la réponse de l’assistant
            labels[i, :start_idx] = -100
            role_ids.append(start_idx)

        batch["labels"] = labels
        return batch


# ==========================================================
# DataCollatorAssistantOnly — version Falcon3-Mamba-7B / LLaMA-style
# ==========================================================
from dataclasses import dataclass
from typing import List, Dict
import torch
from transformers import PreTrainedTokenizerBase

@dataclass
class DataCollatorAssistantOnly:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        # Texte brut depuis ton dataset (colonne "text")
        texts = [f["text"] for f in features]

        # Encodage en batch
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=540,
            return_tensors="pt"
        )

        # On clone les ids pour faire les labels
        labels = batch["input_ids"].clone()

        # Masquage : seules les parties assistant seront entraînées
        for i, ids in enumerate(batch["input_ids"]):
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            start_idx = 0

            # On cherche la position du rôle assistant
            for j, tok in enumerate(tokens):
                # Certains templates contiennent "assistant" ou "<|im_start|>assistant"
                if "assistant" in tok:
                    start_idx = j
                    break

            # Masquer tout avant le rôle assistant
            labels[i, :start_idx] = -100

        batch["labels"] = labels
        return batch
