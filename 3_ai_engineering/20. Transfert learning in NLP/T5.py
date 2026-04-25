from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset


dataset = load_dataset("imdb")

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model_T5 = T5ForConditionalGeneration.from_pretrained("t5-small")

def prepocessing_T5(examples):
    inputs = ['classifier les sentiments : ' + doc for doc in examples["text"]]
    model_input = tokenizer(inputs, max_length=128, truncation=True, padding="max_length" )
    model_input["labels"] = tokenizer(examples["label"], max_length=16, truncation=True, padding="max_length" )("input_ids")
    return model_input


tokenized_T5 = dataset.map(prepocessing_T5, batched=True)

tokenized_T5 = tokenized_T5.remove_columns(["text"])
tokenized_T5 = tokenized_T5.rename_column("label", "labels")
tokenized_T5.set_format("torch")


train_dataset = tokenized_T5["train"]
test_dataset = tokenized_T5["test"]



training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2
)

trainer = Trainer(
    model=model_T5,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
)

trainer.train()

results = trainer.evaluate()
print("Evaluation de T5 :", results)

