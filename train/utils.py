import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from train.train_normal import train_eval
def evaluate(
        net: nn.Module,
        data_loader: DataLoader,
        dtype: torch.dtype,
        device: torch.device,
) -> float:

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(device=device)
            outputs = net(inputs)
            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    return float(correct / total)

#def bo_eval_func(paramters):
