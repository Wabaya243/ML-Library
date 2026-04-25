
# pour la validation on veut pas d'augmentation (utiliser transform_eval). random_split renvoie sous-ensemble avec même transform.
# Remplacer transform du sous-ensemble val pour avoir transform_eval :
# (trick : on peut construire un subset Dataset simple)

# dataset_utils.py
from torch.utils.data import Dataset
from PIL import Image

class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        original_dataset = self.subset.dataset
        original_idx = self.subset.indices[idx]
        img, label = original_dataset.data[original_idx], int(original_dataset.targets[original_idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label
