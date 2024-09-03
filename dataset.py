import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class SignLanguageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label in os.listdir(root_dir):
            label_folder = os.path.join(root_dir, label)
            if os.path.isdir(label_folder):
                for file_name in os.listdir(label_folder):
                    file_path = os.path.join(label_folder, file_name)
                    if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(file_path)
                        self.labels.append(label)

        self.label_to_index = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        label = self.label_to_index[label]

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label).long()

        return image, label
