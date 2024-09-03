from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import SignLanguageDataset

def get_data_loaders(data_folder, batch_size=32, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = SignLanguageDataset(data_folder=data_folder, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return data_loader
