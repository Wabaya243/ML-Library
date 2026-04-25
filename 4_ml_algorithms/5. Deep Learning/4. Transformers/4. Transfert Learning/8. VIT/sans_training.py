# -------------------------
# ViT sur CIFAR-100 - utilisation directe (pas de fine-tuning)
# -------------------------

import torch
import torchvision.transforms as transforms
from transformers import ViTForImageClassification
from PIL import Image

# -------------------------
# 0. Device & seed
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# -------------------------
# 1. Liste des classes CIFAR-100
# -------------------------
classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'cabbage',
    'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
    'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo',
    'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
    'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
    'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain',
    'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
    'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper',
    'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
    'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# -------------------------
# 2. Chargement du modèle ViT pré-entraîné
# -------------------------
model_name = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name, num_labels=100).to(device)
model.eval()  # mode évaluation

# -------------------------
# 3. Transformation pour ViT
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT attend 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# -------------------------
# 4. Prédiction sur une image custom
# -------------------------
def predict_image(img_path):
    """
    Prend une image (chemin), applique les transformations et prédit sa classe avec ViT.
    """
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)  # (1,C,H,W)
    
    with torch.no_grad():
        logits = model(img).logits
        pred_idx = logits.argmax(dim=1).item()  # indice de la classe
        pred_class = classes[pred_idx]          # nom de la classe
    
    return pred_class

# -------------------------
# 5. Exemple d'utilisation
# -------------------------
chemin_de_image = "chemin/vers/image_test.jpg"
print("Classe prédite :", predict_image(chemin_de_image))
