# A Executé que quand j'aurais un tres bon machine 

from unittest import result
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


#Charge le dataset
dataset = load_dataset("pubmed_rct", '2Ok_rct')
print(dataset['train'][0])

#charger Bert
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#tokenizer le dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

#model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-V1.1", num_labels=5)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["validation"],
    tokenizer = tokenizer,
    data_collator = data_collator,
)

trainer.train()

#evaluons les models 
results = trainer.evaluate()
print("Evaluation  de resultat : ", results)


import random 

def augment_text(text):
    synonyms = {
        "cancer": ['tumor', 'malignancy'],
        'study': ['experiment', 'trial'],
        'patient': ['subject', 'case'],
        'treatment': ['therapy', 'procedure'],
        'result': ['outcome', 'effect']}
    words = text.Split()
    new_words = [random.choice(synonyms[word]) if word in synonyms else word for word in words]
    return " ".join(new_words)

augment_data = [augment_text(sample['text']) for sample in dataset["train"]]
print(augment_data[:5])





