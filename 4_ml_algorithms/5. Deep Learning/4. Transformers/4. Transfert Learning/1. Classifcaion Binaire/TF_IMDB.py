# -------------------------
# Étape 0 : Imports
# -------------------------
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset

# -------------------------
# Étape 1 : Charger le dataset IMDB
# -------------------------
datasets = load_dataset("imdb")

# -------------------------
# Étape 2 : Tokenizer BERT
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = datasets.map(tokenize_function, batched=True)

# -------------------------
# Étape 3 : Préparer les données pour TF
# -------------------------
train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=True,
    batch_size=16
)

eval_dataset = tokenized_datasets["test"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=False,
    batch_size=64
)

# -------------------------
# Étape 4 : Charger le modèle TF BERT
# -------------------------
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# -------------------------
# Étape 5 : Compiler le modèle
# -------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ["accuracy"]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# -------------------------
# Étape 6 : Entraînement
# -------------------------
model.fit(train_dataset, validation_data=eval_dataset, epochs=3)

# -------------------------
# Étape 7 : Évaluation
# -------------------------
results = model.evaluate(eval_dataset)
print("Évaluation :", results)

# -------------------------
# Étape 8 : Prédiction
# -------------------------
def predict_sentiment(text):
    encodings = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="tf")
    outputs = model(encodings)
    probs = tf.nn.softmax(outputs.logits, axis=-1)
    pred = tf.argmax(probs, axis=1).numpy()[0]
    sentiment = "positif" if pred == 1 else "négatif"
    print(f"Texte : {text}\nPrédiction : {sentiment}, Probabilités : {probs.numpy()}")

predict_sentiment("I really enjoyed this movie!")
predict_sentiment("The movie was awful and I hated it.")

from transformers import TFAutoModelForCausalLM

# -------------------------
# Étape 9 : Charger GPT-2 en version TensorFlow
# -------------------------
gpt_model = TFAutoModelForCausalLM.from_pretrained("gpt2")

# GPT-2 n’utilise pas le même tokenizer que BERT (mieux vaut en charger un dédié)
from transformers import AutoTokenizer
gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")

# -------------------------
# Étape 10 : Génération de texte
# -------------------------
def generate_text(prompt, max_length=100):
    input_ids = gpt_tokenizer.encode(prompt, return_tensors="tf")
    output = gpt_model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    text = gpt_tokenizer.decode(output[0], skip_special_tokens=True)
    print("Texte généré :", text)
    return text

# Exemple
generate_text("Once upon a time in China")


def sentiment_guided_generation(text):
    # Étape 1 : Prédire le sentiment avec BERT
    encodings = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="tf")
    outputs = model(encodings)
    sentiment = "positive" if tf.argmax(tf.nn.softmax(outputs.logits, axis=-1), axis=1).numpy()[0] == 1 else "negative"
    
    # Étape 2 : Générer du texte avec GPT-2 basé sur ce sentiment
    prompt = f"Write a short story with a {sentiment} mood: {text}"
    input_ids = gpt_tokenizer.encode(prompt, return_tensors="tf")
    generated = gpt_model.generate(input_ids, max_length=100, num_return_sequences=1)
    generated_text = gpt_tokenizer.decode(generated[0], skip_special_tokens=True)
    
    print("Sentiment détecté :", sentiment)
    print("Texte généré :", generated_text)

# Exemple
sentiment_guided_generation("I had a really bad day today")
sentiment_guided_generation("The movie was amazing and I loved it")

