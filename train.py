import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128)
model = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

lambda_sparse = 0.0001
epochs = 10

losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        cls_loss = criterion(outputs, labels)

        sparsity_loss = 0
        for module in model.modules():
            if hasattr(module, "gate_scores"):
                sparsity_loss += torch.sigmoid(module.gate_scores).sum()

        loss = cls_loss + lambda_sparse * sparsity_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    losses.append(running_loss)
    print("Epoch:", epoch+1, "Loss:", running_loss)

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, pred = torch.max(outputs,1)

        total += labels.size(0)
        correct += (pred == labels).sum().item()

acc = 100*correct/total
print("Accuracy:", acc)

total_weights = 0
pruned = 0

for module in model.modules():
    if hasattr(module, "gate_scores"):
        gates = torch.sigmoid(module.gate_scores)

        total_weights += gates.numel()
        pruned += (gates < 1e-2).sum().item()

sparsity = 100 * pruned / total_weights
print("Sparsity:", sparsity)

all_gates = []

for module in model.modules():
    if hasattr(module, "gate_scores"):
        all_gates.extend(torch.sigmoid(module.gate_scores).detach().cpu().numpy().flatten())

plt.hist(all_gates, bins=50)
plt.title("Gate Distribution")
plt.show()
