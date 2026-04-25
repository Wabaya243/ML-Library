# Les Meilleurs variant de Bert et GPT
from ast import arg
from email import message
from os import process_cpu_count
from tokenize import tokenize
from unittest import result
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from dataset import load_dataset

#Charger l'ensemble des données
dataset = load_dataset("ag_news")

#Charger les models Roberta et Tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model_roberta = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4)

#Tokenisons les données
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

#preparer l'ensemble des données
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")


train_dataset = tokenized_datasets["train"]
test_datasets = tokenized_datasets["test"]

#Arguments d'entrainment
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

#Notre entraineur 
trainer = Trainer(
    model=model_roberta,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset = test_datasets,
    processing_class=tokenizer,
)

# Entrainons les model
trainer.train() #Cela vas duré 3 a 4h

#Evaluons les models
results = trainer.evaluate()
print("Evalution du resultat : ", results)



# GPT3 avec la bibliotheque openAI
from openai import OpenAI
client = OpenAI(api_key="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")

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Écris une petite histoire d'un robot qui apprend à cuisiner"}
    ],
    max_tokens=150,
    temperature=0.7
)

print(response.choices[0].message.content.strip())
