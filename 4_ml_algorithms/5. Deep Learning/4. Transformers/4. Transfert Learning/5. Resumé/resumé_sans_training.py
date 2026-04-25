from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "plguillou/t5-base-fr-sum-cnndm"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = """summarize: Les modèles Transformers ont révolutionné le domaine du traitement automatique du langage naturel (NLP).
Ils permettent un traitement parallèle des séquences, améliorant ainsi les performances sur diverses tâches,
comme la traduction automatique, la classification de texte, et la synthèse. Leur architecture basée sur
l’attention permet de mieux capturer les relations contextuelles longues dans les textes.
."""

inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)

print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))




