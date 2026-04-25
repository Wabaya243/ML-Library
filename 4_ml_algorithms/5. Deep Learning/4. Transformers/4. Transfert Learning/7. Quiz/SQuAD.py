# -------------------------
# QA pré-entraîné BERT avec et sans fine-tuning
# -------------------------
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
from torch.utils.data import Dataset
import random

# -------------------------
# 0. Réglages device et seed pour la reproductibilité
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
print("Device utilisé :", device)

# -------------------------
# 1. Dataset toy (SQuAD-style)
# Chaque exemple contient : 
#   - context : le texte dans lequel se trouve la réponse
#   - question : la question à poser
#   - answer : la réponse exacte (span dans context)
# -------------------------
toy_data = [
    {"context": "Paris est la capitale de la France et abrite la Tour Eiffel.",
     "question": "Quelle est la capitale de la France ?",
     "answer": "Paris"},
    {"context": "Albert Einstein était un physicien célèbre. Il a développé la théorie de la relativité.",
     "question": "Qui a développé la théorie de la relativité ?",
     "answer": "Albert Einstein"},
    {"context": "Python est un langage de programmation polyvalent créé par Guido van Rossum dans les années 1990.",
     "question": "Qui a créé Python ?",
     "answer": "Guido van Rossum"}
]

# -------------------------
# 2. Tokenizer et modèle pré-entraîné
# On utilise ici BERT QA déjà fine-tuné sur SQuAD
# -------------------------
MODEL_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"

# Tokenizer
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

# Modèle
model = BertForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)

# -------------------------
# 3. Préparation du dataset pour HuggingFace Trainer
# -------------------------
class QADataset(Dataset):
    """
    Dataset pour QA compatible avec Trainer.
    Encode question + context et calcule start/end positions.
    """
    def __init__(self, examples, tokenizer, max_len=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Tokenisation question + context
        encoding = self.tokenizer(
            ex["question"],
            ex["context"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        # Calcul des indices start/end de la réponse
        answer_text = ex["answer"]
        start_char = ex["context"].find(answer_text)
        end_char = start_char + len(answer_text)
        start_token = encoding.char_to_token(0, start_char)
        end_token = encoding.char_to_token(0, end_char - 1)
        if start_token is None: start_token = 0
        if end_token is None: end_token = 0
        
        # On retire la dimension batch ajoutée par return_tensors
        item = {k: v.squeeze() for k, v in encoding.items()}
        item["start_positions"] = torch.tensor(start_token)
        item["end_positions"] = torch.tensor(end_token)
        return item

train_dataset = QADataset(toy_data, tokenizer)

# -------------------------
# 4. Fine-tuning avec Trainer (optionnel)
# -------------------------
training_args = TrainingArguments(
    output_dir="./qa_finetuned",    # où sauvegarder le modèle
    num_train_epochs=3,             # nombre d'époques
    per_device_train_batch_size=2,  # batch small pour toy dataset
    save_steps=50,
    logging_steps=5,
    learning_rate=3e-5,
    weight_decay=0.01,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)


trainer.train()

# -------------------------
# 5. Fonction de prédiction
# -------------------------
def predict_answer(model, question, context, tokenizer):
    """
    Prédit la réponse à une question donnée dans un contexte.
    """
    model.eval()
    # Tokenisation question+context
    encoding = tokenizer(question, context, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoding)
        # On récupère l'indice du start et end token
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)
        # Reconstruction de la réponse
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(
                encoding["input_ids"][0][start_idx:end_idx+1]
            )
        )
    return answer

# -------------------------
# 6. Test sur le toy dataset (sans fine-tuning)
# -------------------------
print("\n=== Test sur toy dataset (sans fine-tuning) ===")
for ex in toy_data:
    ans = predict_answer(model, ex["question"], ex["context"], tokenizer)
    print(f"Q : {ex['question']}")
    print(f"Predicted answer : {ans}")
    print(f"Gold answer      : {ex['answer']}")
    print("-"*50)

# -------------------------
# 7. Test sur de nouvelles données
# -------------------------
print("\n=== Test sur nouvelles phrases ===")
new_examples = [
    {"context": "Le mont Everest est la plus haute montagne du monde, située dans l'Himalaya.",
     "question": "Quelle est la plus haute montagne du monde ?"},
    {"context": "La Joconde a été peinte par Léonard de Vinci à la Renaissance.",
     "question": "Qui a peint la Joconde ?"},
    {"context": "Le Nil est le plus long fleuve d'Afrique et traverse de nombreux pays.",
     "question": "Quel est le plus long fleuve d'Afrique ?"}
]


for ex in new_examples:
    ans = predict_answer(model, ex["question"], ex["context"], tokenizer)
    print(f"Q : {ex['question']}")
    print(f"Predicted answer : {ans}")
    print("-"*50)

# -------------------------
# 8. Après fine-tuning
# Tu peux réutiliser predict_answer() pour tester les mêmes données
# et observer l'amélioration si tu as fine-tuné le modèle sur ton dataset.
# -------------------------
