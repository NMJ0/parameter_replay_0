import torch
import torch.nn as nn
import torch.nn.functional as F
from avalanche.benchmarks.classic import SplitCIFAR100
from torch.utils.data import DataLoader, Subset


# Fixed CIFAR-100 benchmark: 10 tasks, 10 classes each, seed=42 for reproducibility
benchmark = SplitCIFAR100(n_experiences=10, seed=42, shuffle=False, 
                          class_ids_from_zero_in_each_exp=False)
train_stream = benchmark.train_stream
test_stream = benchmark.test_stream


class BasicBlock(nn.Module):
    """Basic block for ResNet-18"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """ResNet-18 for CIFAR-100"""
    def __init__(self, num_classes=100):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def network_cifar():
    """Create ResNet-18 network for CIFAR-100"""
    return ResNet18(num_classes=100)


def naive_train(model, task_number, epochs, criterion, optimizer, device):
    """Train model on a specific task"""
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

        # Uncomment to see training progress
        # print(f"Task {task_number}, Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")


def test_taskwise(model, task_number, device):
    """Test model on a specific task"""
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
    # print(f"Accuracy on task {task_number}: {acc:.2f}%")
    return acc


def test(model, device):
    """Test model on all tasks and return average accuracy"""
    sum_acc = 0
    acc_list = []
    for i in range(10):  # 10 tasks for CIFAR-100
        acc = test_taskwise(model, i, device)
        sum_acc += acc
        acc_list.append(acc)
    
    avg_acc = sum_acc / 10
    print(f"Average Accuracy: {avg_acc:.2f}%")
    return avg_acc, acc_list


def compute_fisher_information(model, task_number, num_samples, criterion, device):
    """Compute Fisher Information Matrix for a specific task"""
    model.eval()
    experience = train_stream[task_number]
    train_loader = DataLoader(experience.dataset, batch_size=1, shuffle=True)

    # Initialize Fisher information dict (same structure as model params)
    fisher_dict = {name: torch.zeros_like(param, device=device) 
                   for name, param in model.named_parameters()}

    # Limit number of samples for computational efficiency
    count = 0
    for images, labels, *_ in train_loader:
        images, labels = images.to(device), labels.to(device)

        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Accumulate squared gradients (diagonal approximation)
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_dict[name] += param.grad.detach() ** 2

        count += 1
        if count >= num_samples:
            break

    # Average over samples
    for name in fisher_dict:
        fisher_dict[name] /= count

    return fisher_dict


def apply_importance_mask(model, fisher_dict, importance_percent):
    """Apply importance mask to model parameters based on Fisher Information"""
    
    all_importances = torch.cat([v.flatten() for v in fisher_dict.values()])
    k = int(len(all_importances) * (importance_percent / 100.0))

    if k == 0:
        print("Warning: importance_percent too low, no weights kept.")
        k = 1

    # Get threshold for top-k important weights
    threshold = torch.topk(all_importances, k, sorted=True)[0][-1].item()

    mask_dict = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            fisher_vals = fisher_dict[name]
            mask = (fisher_vals >= threshold).float()  # 1 for keep, 0 for prune
            mask_dict[name] = mask
            param.data *= mask  # apply pruning directly

    return model, mask_dict
