# -------------------------
# ViT sur CIFAR-100 avec HuggingFace Trainer
# -------------------------
# Import des librairies
import torch
import torchvision
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, AutoImageProcessor, Trainer, TrainingArguments
from datasets import Dataset as HFDataset
from evaluate import load
import numpy as np
from PIL import Image
import torch.nn as nn

# -------------------------
# 0. Device & seed
# -------------------------
# On définit le device GPU si disponible, sinon CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)  # pour reproductibilité
print(" Device:", device, torch.cuda.get_device_name(0))

# -------------------------
# 1. Préparer CIFAR-100
# -------------------------
# CIFAR-100 est un dataset d'images 32x32 avec 100 classes
# On redimensionne les images à 224x224 pour correspondre à l'entrée attendue par ViT
# On normalise selon les stats des datasets ImageNet (ViT pré-entraîné sur ImageNet-21k)
transform = transforms.Compose([
    transforms.Resize((224,224)),  # ViT attend 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Téléchargement et chargement des datasets
train_ds = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
test_ds  = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

classes = [
    'pomme', 'poisson_d’aquarium', 'bébé', 'ours', 'castor', 'lit', 'abeille', 'scarabée',
    'vélo', 'bouteille', 'bol', 'garçon', 'pont', 'bus', 'papillon', 'chou',
    'chameau', 'canette', 'château', 'chenille', 'bétail', 'chaise', 'chimpanzé', 'horloge',
    'nuage', 'cafard', 'canapé', 'crabe', 'crocodile', 'tasse', 'dinosaure', 'dauphin',
    'éléphant', 'poisson_plat', 'forêt', 'renard', 'fille', 'hamster', 'maison', 'kangourou',
    'clavier', 'lampe', 'tondeuse', 'léopard', 'lion', 'lézard', 'homard', 'homme',
    'érable', 'moto', 'montagne', 'souris', 'champignon', 'chêne', 'orange',
    'orchidée', 'loutre', 'palmier', 'poire', 'camionnette', 'pin', 'plaine',
    'assiette', 'coquelicot', 'porc-épic', 'opossum', 'lapin', 'raton_laveur', 'raie', 'route',
    'fusée', 'rose', 'mer', 'phoque', 'requin', 'musaraigne', 'moufette', 'gratte_ciel',
    'escargot', 'serpent', 'araignée', 'écureuil', 'tramway', 'tournesol', 'poivron',
    'table', 'char', 'téléphone', 'télévision', 'tigre', 'tracteur', 'train', 'truite',
    'tulipe', 'tortue', 'armoire', 'baleine', 'saule', 'loup', 'femme', 'ver'
]


r"""
# -------------------------
# 2. Conversion en datasets HuggingFace
# -------------------------
# Trainer HuggingFace fonctionne mieux avec des datasets de type `datasets.Dataset`
def generator(dataset):
    for img, label in dataset:
        yield {
            "pixel_values": torch.tensor(np.array(img), dtype=torch.float32), # convertie en tableau numpy car hf adore ça
            "labels": int(label)
        }

hf_train_ds = HFDataset.from_generator(lambda: generator(train_ds))
hf_test_ds  = HFDataset.from_generator(lambda: generator(test_ds))

"""
# -------------------------
# 3. Charger ViT pré-entraîné
# -------------------------
# On utilise le ViT pré-entraîné sur ImageNet-21k
model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = AutoImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, num_labels=100).to(device)
# num_labels=100 car CIFAR-100 contient 100 classes

"""
# -------------------------
# 4. Préparer fonction de preprocessing pour Trainer
# -------------------------
# Trainer s'attend à recevoir des tensores torch pour "pixel_values" et "labels"
# HuggingFace gère la conversion en tenseurs automatiquement
# pas besoin de map ni de torch.tensor()


hf_train_ds.set_format(type="torch", columns=["pixel_values", "labels"])
hf_test_ds.set_format(type="torch", columns=["pixel_values", "labels"])
"""

# -------------------------
# 5. Metrics (accuracy)
# -------------------------
# load_metric("accuracy") est fourni par HuggingFace pour calculer la précision
metric = load("accuracy")

# Fonction pour calculer l'accuracy lors de l'évaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

def torch_collator(batch):
    pixel_values = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    return {"pixel_values": pixel_values, "labels": labels}


# -------------------------
# 6. TrainingArguments
# -------------------------
# HuggingFace Trainer utilise TrainingArguments pour gérer le training loop
training_args = TrainingArguments(
    output_dir="./vit_cifar100_trainer",    # répertoire de sauvegarde
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    eval_strategy="epoch",           # évaluer à chaque epoch
    save_strategy="epoch",                 # sauvegarder modèle à chaque epoch
    logging_strategy="steps",              
    logging_steps=100,                     # logs tous les 100 steps
    learning_rate=7e-5,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    save_total_limit=2,                    # garder seulement les 2 derniers modèles
    load_best_model_at_end=True,           # recharger le meilleur modèle selon la metric
    metric_for_best_model="accuracy",
    fp16=True,
)

#  Geler le backbone ViT (ne pas réentraîner les couches pré-entraînées)
for name, param in model.vit.named_parameters():
    if "encoder.layer.10" in name or "encoder.layer.11" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Remplacer la tête par un petit MLP custom
if hasattr(model, "classifier"):
    model.classifier = nn.Sequential(
        nn.Linear(model.config.hidden_size, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 100)
    ).to(device)
else:
    model.classifier = nn.Sequential(
        nn.Linear(model.config.hidden_size, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 100)
    ).to(device)



# -------------------------
# 7. Définir Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=torch_collator,
    compute_metrics=compute_metrics
)

# -------------------------
# 8. Lancer le training
# -------------------------


torch.cuda.empty_cache()
trainer.train()

#sauvegarde le modèle
trainer.save_model("Save/vit_cifar100_trainer")
feature_extractor.save_pretrained("Save/vit_cifar100_trainer")


# -------------------------
# 9. Évaluation finale
# -------------------------
results = trainer.evaluate()
print("Test Accuracy:", results["eval_accuracy"])

# -------------------------
# 10. Tester sur image custom
# -------------------------
def predict_image(img_path):
    """
    Prend une image (chemin), applique les transformations et prédit sa classe avec ViT.
    """
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # (1,C,H,W)
    img = img.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(img).logits
        pred = logits.argmax(dim=1).item()
        pred_class = classes[pred]  # mapping indice → nom de classe
   
    return pred_class



r"""

model = ViTForImageClassification.from_pretrained("Save/vit_cifar100_trainer")
feature_extractor = AutoImageProcessor.from_pretrained("Save/vit_cifar100_trainer")


# Exemple d'utilisation :
chemin_de_image = "chemin/vers/image_test.jpg"
    
print("Classe prédite:", predict_image(chemin_de_image))

"""





