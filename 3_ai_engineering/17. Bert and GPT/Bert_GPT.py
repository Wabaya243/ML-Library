# Attention is All You Need


from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Charger le dataset IMDB
datasets = load_dataset("imdb")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # (tu avais écrit 'bert-case-uncased')

# Tokenisation de l'ensemble des données
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = datasets.map(tokenize_function, batched=True)

# Préparer les données
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")  # (tu avais écrit "renome_columns")
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# Charger le modèle pré-entraîné
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


# Definir les arguments d'entrainement
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)


trainer.train()

#evaluer les models 
results = trainer.evaluate()
print("Evaluation des mes resultats", results)

#Experimenter avec GPT
from transformers import AutoModelForCausualLM

gpt_model = AutoModelForCausualLM.from_pretrained("gpt2")

input_text = "Once Upon Time In China"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = gpt_model.generate(input_ids, max_length=50, num_return_sequences=1)

print("Generated_Text : ", tokenizer.decode(output[0], skip_special_token=True))