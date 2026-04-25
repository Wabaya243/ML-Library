#Préparation du dataset
from random import triangular
from datasets import load_dataset

dataset = load_dataset('wikitext', "wikitext-2-raw-v1")
print(dataset)

##### Tokenization pour MLM
#On prépare des tokens masqués aléatoirement.
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_mlm(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Appliquer la tokenization et supprimer la colonne text
tokenized_datasets = dataset.map(tokenize_mlm, batched=True, remove_columns=["text"])

##### Préparer le masque aléatoire
#HuggingFace propose DataCollatorForLanguageModeling qui fait ça automatiquement :

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15, # 15% des token seront masqué
)

##### Charger le modèle BERT pour MLM

from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

##### Fine-tuning
from transformers import Trainer, TrainingArguments
import torch

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    mask = labels != -100  # on ne compte que les tokens masqués
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    correct = (predictions[mask] == torch.tensor(labels)[mask]).sum().item()
    total = mask.sum().item()
    accuracy = correct / total
    return {"accuracy": accuracy}


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results_mlm',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    save_steps=500,                 # on ne sauvegarde pas trop souvent
    logging_dir='./logs_mlm',
    logging_steps=20,               # affiche la loss d’entraînement tous les 20 batchs
    evaluation_strategy="steps",    # on évalue sur validation pendant l’entraînement
    eval_steps=100,                 # toutes les 100 batchs → metrics affichés
    remove_unused_columns=False,
    load_best_model_at_end=True,    # garde le meilleur modèle sur validation
    metric_for_best_model="eval_loss"
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

### Test
# Exemple de texte avec token masqué
text = "The movie was [MASK] and I really enjoyed it."

# Tokenization
import torch

inputs = tokenizer(text, return_tensors="pt")
mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]

#prediction 
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

#recuper la prediction du token masqué
mask_token_logits = logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

print("Top 5 des predictions pour [MASK] :")
for token in top_5_tokens:
    word = tokenizer.decode([token])
    print(word)




