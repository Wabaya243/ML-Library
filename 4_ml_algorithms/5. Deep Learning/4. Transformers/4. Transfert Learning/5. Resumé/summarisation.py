from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import evaluate
import torch
import os
from transformers import DataCollatorForSeq2Seq

# =========================
# 1. CHARGEMENT DU DATASET
# =========================
try:
    # Charger le dataset WikiLingua en français (déjà format parquet, prêt à l'emploi)
    dataset = load_dataset("esdurmus/wiki_lingua", "french")
except Exception as e:
    print("Erreur lors du chargement du dataset :", e)
    exit()


print(dataset["train"][0].keys())  # clés du dict 'article' ou autres colonnes
print(dataset["train"][0]["article"].keys())


# =========================
# 2. CHARGEMENT DU TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained('t5-small')

# =========================
# 3. FONCTION DE TOKENISATION
# =========================

# Fonction qui transforme un batch d'exemples du dataset en tokens utilisables par le modèle
def tokenize_function(examples):
    inputs = []   # liste pour stocker les textes d'entrée
    targets = []  # liste pour stocker les résumés / labels

    # Parcourir chaque article du batch (examples["article"] est une liste de dicts)
    for article in examples["article"]:
        doc = article["document"]  # récupérer le texte principal

        # Si le document est une liste de phrases, on les concatène en un seul texte
        if isinstance(doc, list):
            text = " ".join(doc)
        else:
            text = doc
        
        # Préfixe 'summarize:' utilisé pour T5 pour indiquer la tâche
        inputs.append("summarize: " + text)

        # Récupérer le résumé associé, si existant
        summ = article.get("summary", "")
        if isinstance(summ, list):  # si c'est une liste, on concatène aussi
            summaries = " ".join(summ)
        else:
            summaries = summ or ""  # sinon on prend la string ou une string vide
        targets.append(summaries)

    # -----------------------------
    # Tokenisation des entrées
    # -----------------------------
    # tokenizer convertit les textes en listes d'IDs de tokens
    # max_length=512 tronque les textes trop longs, padding=False => pas de padding ici
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=False)

    # -----------------------------
    # Tokenisation des labels / cibles
    # -----------------------------
    # Utilisation du contexte as_target_tokenizer pour que T5 gère correctement les labels
    with tokenizer.as_target_tokenizer():
        # max_length=150 pour les résumés, padding forcé à cette longueur
        labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length")

    # -----------------------------
    # Préparer les labels pour la loss
    # -----------------------------
    # Remplacer le pad_token_id par -100, car PyTorch ignore les -100 dans la loss
    label_ids = labels["input_ids"]
    label_ids = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
        for seq in label_ids
    ]

    # Ajouter les labels au dictionnaire des entrées du modèle
    model_inputs["labels"] = label_ids

    # Retourner le dictionnaire complet prêt pour l'entraînement
    return model_inputs



# Tokenisation du dataset
# map (batched=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# =========================
# 4. CHARGEMENT DU MODÈLE
# =========================
model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')


# --- data collator seq2seq (gère labels et padding dynamiquement) ---
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)


# -----------------------------
# Créer les dossiers de sortie et de logs si inexistants
# -----------------------------
# ./results → dossier où seront sauvegardés les checkpoints du modèle
# ./logs    → dossier où seront écrits les logs pour TensorBoard
os.makedirs("./results", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

# Chemins absolus pour plus de sécurité et compatibilité
logging_dir = os.path.abspath("./logs")
output_dir = os.path.abspath("./results")

# -----------------------------
# Paramètres d'entraînement avec HuggingFace Transformers
# -----------------------------
training_args = TrainingArguments(
    output_dir=output_dir,       # dossier de sauvegarde des checkpoints
    eval_strategy="steps",       # évaluation du modèle à la fin de chaque epoch
    eval_steps=100,
    save_strategy="steps",       # sauvegarde tous les N steps
    save_steps=500,              # fréquence de sauvegarde en nombre de steps
    logging_dir=logging_dir,     # dossier pour logs TensorBoard
    logging_strategy="steps",    # fréquence de logs
    logging_steps=50,            # tous les 50 steps
    learning_rate=2e-5,          # learning rate pour l'optimiseur
    per_device_train_batch_size=16,  # batch size pour l'entraînement
    per_device_eval_batch_size=16,   # batch size pour l'évaluation
    num_train_epochs=3,          # nombre d'epochs d'entraînement
    weight_decay=0.01,           # régularisation L2
    save_total_limit=2,          # nombre maximum de checkpoints à garder
    report_to="none",            # désactive reporting vers TensorBoard ou W&B
    no_cuda=not torch.cuda.is_available(),  # utilise CPU si CUDA indisponible
)



# =========================
# 6. CRÉATION DU TRAINER
# =========================

# -----------------------------
# Vérifier si le dataset contient un split validation
# -----------------------------
if "validation" not in tokenized_datasets and "test" not in tokenized_datasets:
    # Si aucun split validation ou test n'existe,
    # on crée un split manuellement à partir du train (ici 90% train / 10% validation)
    tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1)
    train_dataset = tokenized_datasets["train"]  # 90% des données → train
    eval_dataset = tokenized_datasets["test"]    # 10% restante → validation
else:
    # Si validation ou test déjà présent dans le dataset
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"] if "validation" in tokenized_datasets else tokenized_datasets["test"]


# -----------------------------
# Création du Trainer HuggingFace
# -----------------------------
trainer = Trainer(
    model=model,                    # modèle à entraîner
    args=training_args,             # arguments d'entraînement définis avant
    train_dataset=train_dataset,    # dataset d'entraînement
    eval_dataset=eval_dataset,      # dataset d'évaluation
    tokenizer=tokenizer,            # tokenizer pour la conversion texte→IDs
    data_collator=data_collator,    # gère le padding dynamique et les labels pour Seq2Seq
)


# -----------------------------
# Suppression du callback TensorBoard (optionnel)
# -----------------------------
# Certains setups ajoutent automatiquement TensorBoardCallback, ici on le retire
try:
    trainer.remove_callback(TensorBoardCallback)
except Exception:
    # Si la méthode ci-dessus échoue, on filtre simplement la liste des callbacks
    trainer.callback_handler.callbacks = [
        cb for cb in trainer.callback_handler.callbacks
        if cb.__class__.__name__ != "TensorBoardCallback"
    ]



# =========================
# 7. ENTRAÎNEMENT
# =========================

trainer.train()


# =========================
# 8. SAUVEGARDE DU MODÈLE
# =========================
save_dir = "./trained_model"
os.makedirs(save_dir, exist_ok=True)

model.save_pretrained(save_dir)       # sauvegarde des poids du modèle
tokenizer.save_pretrained(save_dir)   # sauvegarde du tokenizer
print(f"✅ Modèle et tokenizer sauvegardés dans : {save_dir}")



# =========================
# 9. TEST MANUEL DU MODÈLE
# =========================
sample_text = """
Les modèles Transformers ont révolutionné le domaine du traitement automatique du langage naturel (NLP).
Ils permettent un traitement parallèle des séquences, améliorant ainsi les performances sur diverses tâches,
comme la traduction automatique, la classification de texte, et la synthèse. Leur architecture basée sur
l’attention permet de mieux capturer les relations contextuelles longues dans les textes.
"""


inputs = tokenizer("summarize: " + sample_text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)


print("Synthèse générée :", tokenizer.decode(outputs[0], skip_special_tokens=True))

# =========================
# 10. ÉVALUATION AUTOMATIQUE
# =========================

# Charger la métrique ROUGE (utile pour évaluer la qualité des résumés générés)
metric = evaluate.load('rouge')

# Listes pour stocker les prédictions du modèle et les références (résumés corrects)
predictions = []
references = []

# Choisir le dataset d'évaluation : validation si elle existe, sinon test
eval_dataset = tokenized_datasets["validation"] if "validation" in tokenized_datasets else tokenized_datasets["test"]

# Boucle sur chaque exemple du dataset d'évaluation
for example in eval_dataset:
    # Transformer les input_ids en tenseur PyTorch et ajouter une dimension batch (B=1)
    input_ids = torch.tensor(example["input_ids"]).unsqueeze(0)

    # Générer le résumé avec le modèle seq2seq
    # max_length=150 → longueur maximale du résumé
    # num_beams=4 → beam search pour améliorer la qualité
    # early_stopping=True → stop la génération si tous les beams se terminent
    generated_ids = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)

    # Décoder les IDs générés en texte lisible et ignorer les tokens spéciaux (<pad>, <eos>)
    pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    predictions.append(pred)  # ajouter la prédiction à la liste

    # Nettoyage des labels : remplacer -100 (padding) par rien
    label_ids = [id for id in example["labels"] if id != -100]
    # Décoder les labels nettoyés en texte lisible
    ref = tokenizer.decode(label_ids, skip_special_tokens=True)
    references.append(ref)  # ajouter la référence à la liste

# Calcul des scores ROUGE en comparant les prédictions et les références
# use_stemmer=True → réduction des mots à leur racine pour comparer plus efficacement
results = metric.compute(predictions=predictions, references=references, use_stemmer=True)

# Affichage des scores ROUGE (ex. ROUGE-1, ROUGE-2, ROUGE-L)
print("Scores ROUGE :", results)


####################
######### pour charger les models ##################

# Chemin vers le dossier où le modèle et le tokenizer ont été sauvegardés
save_dir = "./trained_model"

# Charger le tokenizer depuis le dossier sauvegardé
# Le tokenizer permet de transformer du texte en IDs numériques utilisables par le modèle
tokenizer = AutoTokenizer.from_pretrained(save_dir)

# Charger le modèle seq2seq (T5 ou autre) depuis le dossier sauvegardé
# Le modèle contient les poids entraînés sur ton dataset
model = AutoModelForSeq2SeqLM.from_pretrained(save_dir)

# =========================
# Test rapide du modèle
# =========================

# Texte d'exemple à résumer
text = """ summarize: Les modèles Transformers ont révolutionné le domaine du traitement automatique du langage naturel (NLP).
Ils permettent un traitement parallèle des séquences, améliorant ainsi les performances sur diverses tâches,
comme la traduction automatique, la classification de texte, et la synthèse. Leur architecture basée sur
l’attention permet de mieux capturer les relations contextuelles longues dans les textes.
."""

# Tokenisation du texte d'entrée : transforme le texte en tenseur PyTorch d'IDs
inputs = tokenizer(text, return_tensors="pt")

# Génération du résumé par le modèle
# max_length=150 → limite la longueur du résumé généré
outputs = model.generate(**inputs, max_length=150)

# Décodage des IDs générés en texte lisible
# skip_special_tokens=True → supprime les tokens spéciaux comme <pad> ou <eos>
print(tokenizer.decode(outputs[0], skip_special_tokens=True))











