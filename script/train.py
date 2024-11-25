import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# Dataset paths
train_dir = "dataset/train"
val_dir = "dataset/val"

# Transforms for training and validation
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]),
}

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms["train"])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms["val"])

# DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# RCNN Model Definition
class RCNN(nn.Module):
    def __init__(self, num_classes):
        super(RCNN, self).__init__()
        # CNN backbone for feature extraction
        self.cnn = resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])  # Remove the classifier

        # RNN for sequence processing
        self.rnn = nn.LSTM(input_size=512, hidden_size=128, num_layers=1, batch_first=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # CNN feature extraction
        batch_size = x.size(0)
        cnn_features = self.cnn(x)  # [B, 512, H, W]
        cnn_features = cnn_features.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 512]
        cnn_features = cnn_features.view(batch_size, -1, 512)  # Flatten spatial dimensions

        # RNN processing
        rnn_out, _ = self.rnn(cnn_features)

        # Use the last output of the RNN
        rnn_out = rnn_out[:, -1, :]

        # Classification
        out = self.fc(rnn_out)
        return out

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_dataset.classes)
model = RCNN(num_classes=num_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Training Loss: {running_loss/len(train_loader):.4f}")
        validate_model(model, val_loader, criterion)

# Validation loop
def validate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Loss: {running_loss/len(val_loader):.4f}")
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# Train and save the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
torch.save(model.state_dict(), "rcnn_plant_disease.pth")
print("Model saved as rcnn_plant_disease.pth")
