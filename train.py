import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SignLanguageDataset
import torch.nn.functional as F  # Add this import

class SignLanguageModel(nn.Module):
    def __init__(self, num_classes=36):  # Adjust num_classes as needed
        super(SignLanguageModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # Adjust size based on your data
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)  # Adjust size based on your data
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_train_loader():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = SignLanguageDataset(root_dir='sign_language_data', transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    return train_loader

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignLanguageModel(num_classes=36).to(device)
    train_loader = get_train_loader()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = (correct_predictions / total_samples) * 100

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    torch.save(model.state_dict(), 'sign_language_model.pth')
    print('Model saved to sign_language_model.pth')

if __name__ == '__main__':
    train()
