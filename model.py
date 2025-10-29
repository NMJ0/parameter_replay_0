import torch
import torch.nn as nn
import torch.nn.functional as F

from avalanche.benchmarks.classic import SplitMNIST
from torch.utils.data import DataLoader, Subset
import copy

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
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x  

    return MLP()

def naive_train(model,task_number, epochs,criterion,optimizer,device):
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
def ewc_train(model,task_number, epochs,criterion,optimizer, fisher_dict_prev, parameter_dict_prev, ewc_lambda, device):
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

            # EWC regularization
            if len(fisher_dict_prev) > 0 and len(parameter_dict_prev) > 0:
                ewc_loss=0
                for i in range(task_number):
                    fisher_dict = fisher_dict_prev[i]
                    optpar_dict = parameter_dict_prev[i]
                    for name, param in model.named_parameters():
                        if name in fisher_dict:
                            fisher = fisher_dict[name]
                            optpar = optpar_dict[name]
                            ewc_loss += (fisher * (param - optpar).pow(2)).sum()
                loss += (ewc_lambda / 2) * ewc_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        #print(f"EWC Task {task_number}, Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

def test_taskwise(model,task_number,device):
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
    sum=0
    acc_list=[]
    for i in range(5):
        acc=test_taskwise(model,i,device)
        sum+=acc
        acc_list.append(acc)
    acc = sum / 5
    print(f"Average Accuracy: {acc:.2f}%")
    return acc,acc_list




def compute_fisher_information(model, task_number, num_samples, criterion, device):
    model.eval()
    experience = train_stream[task_number]
    train_loader = DataLoader(experience.dataset, batch_size=1, shuffle=True)

    # Initialize Fisher information dict (same structure as model params)
    fisher_dict = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}

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


def create_masked_weight_dict(model, mask_dict):
    device = next(model.parameters()).device
    masked_weight_dict = copy.deepcopy(model.state_dict())
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask_dict:
                mask = mask_dict[name].to(device)
                masked_weight_dict[name] = param.data * mask   
                             
    return masked_weight_dict


def load_non_zero_weights(model, weight_dict):
    device = next(model.parameters()).device
    with torch.no_grad():
        # Iterate over the model's current parameters
        for name, param in model.named_parameters():
            if name in weight_dict:
                saved_tensor = weight_dict[name].to(device)
                mask = (saved_tensor != 0)
                param.data[mask] = saved_tensor[mask]
                
    model.eval()
    return model