from random import shuffle
import torch 
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

#charger les données
dataset = load_dataset("imdb")
train_texts, train_labels, test_texts, test_labels = train_test_split(dataset["train"]["text"], dataset["train"]["label"], test_size=0.2, random_state=42)

#Tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenizer_data(texts, labels, tokennizer, max_length=128):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    return {
        "input_ids" : encodings['input_ids'],
        "attetnion_mask": encodings["attention_mask"],
        'labels': labels
    }

train_data = tokenizer_data(train_texts, train_labels, tokenizer)
test_data = tokenizer_data(test_texts, test_labels, tokenizer)


class IMDBDataset(Dataset):
    def __init__(self, data):
        self.input_ids = data["input_ids"] 
        self.attention_mask = data["attetnion_mask"]
        self.labels = data["labels"]
    def __len__(self):
        return len(self.labels)

    def __getItem__(self, idx):
        return {
            'input_ids' : torch.tensor(self.input_ids[idx]),
            'attention_mask' : torch.tensor(self.attention_mask[idx]),
            'labels' : torch.tensor(self.labels[idx])
        }

train_dataset = IMDBDataset(train_data)
test_dataset = IMDBDataset(test_data)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


#Charger les models pre_entrainé 
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
num_training_steps =len(train_loader) * 3  ## le nombre d'epoch 3
warmup_steps = int(0.1 * num_training_steps)

scheduler = get_scheduler(
    "slanted_triangular",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps,
)

# la boucle d'entrainement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_model():
    model.train()
    for epoch in range(3):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    print("Entrainement terminé")

train_model()

# évaluation du modèle
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = logits.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())
f1 = f1_score(all_labels, all_preds, average="weighted")
print(f"F1 Score: {f1}")


from sacrebleu import BLEU

references = [['this is a test samples','this is a test examples']]
hypothesis = ['this is a test sample']

bleu = BLEU()
score = bleu.corpus_score(hypothesis, references)
print(f"BLEU Score: {score}")






