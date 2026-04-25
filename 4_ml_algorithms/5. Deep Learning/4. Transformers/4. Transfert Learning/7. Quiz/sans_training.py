# -------------------------
# QA pré-entraîné BERT - test direct sans fine-tuning
# -------------------------
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering

# -------------------------
# 0. Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilisé :", device)

# -------------------------
# 1. Charger le modèle BERT pré-entraîné pour QA
# -------------------------
MODEL_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model = BertForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)

# -------------------------
# 2. Fonction de prédiction
# -------------------------
def predict_answer(model, question, context, tokenizer):
    """
    Prédit la réponse à une question dans un contexte donné.
    """
    model.eval()
    encoding = tokenizer(question, context, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoding)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(
                encoding["input_ids"][0][start_idx:end_idx+1]
            )
        )
    return answer

# -------------------------
# 3. Exemples de test
# -------------------------
test_examples = [
    {"context": "Paris est la capitale de la France et abrite la Tour Eiffel.",
     "question": "Quelle est la capitale de la France ?"},
    {"context": "Albert Einstein était un physicien célèbre. Il a développé la théorie de la relativité.",
     "question": "Qui a développé la théorie de la relativité ?"},
    {"context": "Le mont Everest est la plus haute montagne du monde, située dans l'Himalaya.",
     "question": "Quelle est la plus haute montagne du monde ?"},
    {"context": "La Joconde a été peinte par Léonard de Vinci à la Renaissance.",
     "question": "Qui a peint la Joconde ?"}
]

# -------------------------
# 4. Affichage des prédictions
# -------------------------
print("\n=== Résultats sans fine-tuning ===")
for ex in test_examples:
    ans = predict_answer(model, ex["question"], ex["context"], tokenizer)
    print(f"Q: {ex['question']}")
    print(f"Predicted answer: {ans}")
    print("-"*50)
