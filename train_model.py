import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
import torch.nn as nn
import torch.optim as optim

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# Dataset
dataset = datasets.ImageFolder("dataset", transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# ✅ MobileNet Model (SMALL SIZE)
model = models.mobilenet_v2(pretrained=True)

# change last layer
model.classifier[1] = nn.Linear(1280, 2)

# Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
epochs = 5

for epoch in range(epochs):
    running_loss = 0

    for images, labels in train_loader:

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

# Save
torch.save(model, "lung_model.pth")

print("🔥 Model Saved")