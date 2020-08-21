import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from src.model import resnet18
from torch.utils.data import DataLoader
from torchvision import datasets
from src.utils import save_checkpoint_classifier

device = 'cuda' if torch.cuda.is_available else 'cpu'

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])])

train_dataset = datasets.STL10(
    './data', split='train', download=False, transform=data_transform)
train_loader = DataLoader(
    train_dataset, batch_size=128, num_workers=8)

model = resnet18().to(device)

### Train ###
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 50

for epoch in range(epochs):
    t = time.time()
    for x, label in train_loader:
        x, label = x.to(device), label.to(device)

        preds = model(x)  # [batch, num_classes]
        loss = criterion(preds, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("[%d/%d] baseline loss : %.4f | time : %.2fs" %
            (epoch + 1, epochs, loss.item(), time.time() - t))

save_checkpoint_classifier(model, 'checkpoints/baseline.pt')


### Test ###
test_dataset = datasets.STL10(
    './data', split='test', download=False, transform=data_transform)
test_loader = DataLoader(
    test_dataset, batch_size=128, num_workers=8)

with torch.no_grad():
    correct, total = 0, 0
    for x, label in test_loader:
        x, label = x.to(device), label.to(device)
        
        output = model(x)
        preds = torch.argmax(output, dim=1)

        correct += int(torch.sum(preds == label))
        total += int(label.size(0))

    print('Test Accuracy : %.4f'%(correct / total))
