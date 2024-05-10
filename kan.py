!ls C:\JPMC\qds\NLP\SmartBuzz\efficient-kan\src

import sys 

sys.path.append(r'C:\JPMC\qds\NLP\SmartBuzz\efficient-kan\src')

from efficient_kan import KAN

# Train on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load MNIST
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
valset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

# Define model
model = KAN([28 * 28, 64, 10])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# Define loss
criterion = nn.CrossEntropyLoss()
for epoch in range(10):
    # Train
    model.train()
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.view(-1, 28 * 28).to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

    # Validation
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in valloader:
            images = images.view(-1, 28 * 28).to(device)
            output = model(images)
            val_loss += criterion(output, labels.to(device)).item()
            val_accuracy += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    val_loss /= len(valloader)
    val_accuracy /= len(valloader)

    # Update learning rate
    scheduler.step()

    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
    )

from efficient_kan import KANLinear

class KAN_AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            KANLinear(in_features = 784, out_features = 256),
            # KANLinear(in_features = 784, out_features = 1024),
            # KANLinear(in_features = 1024, out_features = 512),
            # KANLinear(in_features = 512, out_features = 256),
            KANLinear(in_features = 256, out_features = n_comp)
         )
        self.decoder = nn.Sequential(
            KANLinear(in_features = n_comp, out_features = 256),
            # KANLinear(in_features = 256, out_features = 512),
            # KANLinear(in_features = 512, out_features = 1024),
            # KANLinear(in_features = 1024, out_features = 784)
            KANLinear(in_features = 256, out_features = 784)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        x = torch.flatten(encoded, 1)
        decoded = self.decoder(x)
        return encoded, decoded

model = KAN_AutoEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

def train(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print("Epoch started!")
        with tqdm(dataloader) as pbar:
            for data in dataloader:
                img, _ = data
                img = img.view(img.size(0), -1) # Flatten
                optimizer.zero_grad()
                encoded, decoded = model(img)
                loss = criterion(decoded, img)
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

num_epochs = 10
train(model, trainloader, criterion, optimizer, num_epochs)

def evaluate_model(model, dataloader, criterion):
    model.eval() # Set model to evaluation mode
    total_loss = 0
    with torch.no_grad(): # No gradients needed
        for data in dataloader:
            img, _ = data
            img = img.view(img.size(0), -1)
            _, decoded = model(img)
            loss = criterion(decoded, img)
            total_loss += loss.item()
    
    average_loss = total_loss / len(dataloader)
    print(f'Average Reconstruction Loss: {average_loss:.4f}')
    return average_loss

evaluate_model(model, trainloader, criterion)
