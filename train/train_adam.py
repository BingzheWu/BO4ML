import torch
import sys
sys.path.append("train")
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import torch.optim as optim
from train.utils import evaluate

def train_eval(net: torch.nn.Module,
          train_loader: DataLoader,
          test_loader: DataLoader,
          parameters: Dict,
          dtype: torch.dtype,
          device: torch.device):

    net.to(dtype=dtype, device=device)
    net.train()

    ## define loss
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    beta1 = parameters.get("beta1", 0.9)
    beta2 = parameters.get("beta2", 0.999)
    weight_decay = parameters.get("weight_decay", 0.0)
    optimizer = optim.Adam(net.parameters(),
                           lr = parameters.get('lr', 0.0001),
                           betas = (beta1, beta2),
                           weight_decay = weight_decay
                           )
    num_epochs = parameters.get("num_epochs", 30)

    for _ in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(device=device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if _%5==0:
            acc = evaluate(net, test_loader, dtype, device)
            net.train()
            print("Acc of epoch {} is {}".format(_, acc))
    return acc
