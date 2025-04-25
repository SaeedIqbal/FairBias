import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

DATASET_PATHS = {
    "CheXpert": {
        "images_dir": "/data/CheXpert/images/",
        "labels_file": "/data/CheXpert/labels.csv",
    },
    "Breast_MRI_FFD": {
        "images_dir": "/data/Breast_MRI_FFD/images/",
        "labels_file": "/data/Breast_MRI_FFD/labels.csv",
    },
    "Hep2": {
        "images_dir": "/data/Hep2/images/",
        "labels_file": "/data/Hep2/labels.csv",
    },
    "SOKL": {
        "images_dir": "/data/SOKL/images/",
        "labels_file": "/data/SOKL/labels.csv",
    },
}

class MedicalDataset(Dataset):
    def __init__(self, dataset_name, transform=None):
        self.dataset_name = dataset_name
        self.transform = transform

        # Load labels from CSV file
        labels_path = DATASET_PATHS[dataset_name]["labels_file"]
        self.labels_df = pd.read_csv(labels_path)

        # Extract image paths and labels
        self.image_paths = [
            f"{DATASET_PATHS[dataset_name]['images_dir']}{filename}"
            for filename in self.labels_df["filename"]
        ]
        self.labels = self.labels_df["label"].values  # Assuming 'label' column contains one-hot encoded labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label