import torch
import torch.nn as nn
import torch.nn.functional as F

from avalanche.benchmarks.classic import SplitMNIST
from torch.utils.data import DataLoader, Subset


benchmark = SplitMNIST(n_experiences=5, seed=1,shuffle=False)
train_stream = benchmark.train_stream
test_stream = benchmark.test_stream

def network_mnist(size_first_layer, size_second_layer):
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, size_first_layer)
            self.fc2 = nn.Linear(size_first_layer, size_second_layer)
            self.fc3 = nn.Linear(size_second_layer, 10)

        def forward(self, x):
            # Flatten the input: (batch_size, 1, 28, 28) -> (batch_size, 784)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x  # logits (use CrossEntropyLoss, so no softmax here)

    return MLP()

def naive_train(model,task_number, epochs,criterion,optimizer,device):
    """Train the model on a given SplitMNIST task."""
    experience = train_stream[task_number]
    train_loader = DataLoader(experience.dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels, *_ in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        #print(f"Task {task_number}, Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

def test_taskwise(model,task_number,device):
    """Test the model only on a specific SplitMNIST task."""
    experience = test_stream[task_number]
    test_loader = DataLoader(experience.dataset, batch_size=64, shuffle=False)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels, *_ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Accuracy on task {task_number}: {acc:.2f}%")
    return acc

def test(model,device):
    """Test the model on all tasks (0–4) combined."""
    sum=0
    acc_list=[]
    for i in range(5):
        acc=test_taskwise(model,i,device)
        sum+=acc
        acc_list.append(acc)
    acc = sum / 5
    print(f"Average Accuracy: {acc:.2f}%")
    return acc,acc_list
