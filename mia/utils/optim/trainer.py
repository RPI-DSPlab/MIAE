# Define the model architecture
from copy import deepcopy

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from utils.models.loader import init_network


# Define EMA
class EMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(EMA, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


# Training function
def train(model, device, train_loader, optimizer, scheduler=None):
    model.train()
    ema = EMA(model, 0.999)
    weight_decay = 0.01

    scaler = GradScaler()  # for mixed precision training
    running_loss = 0.0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # forward
        with autocast():
            output = model(data)
            cross_entropy_loss = nn.CrossEntropyLoss()(output, target)

            # weight decay loss
            l2_reg = torch.tensor(0., requires_grad=True)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l2_reg = l2_reg + torch.norm(param, 2)
            loss_wd = 0.5 * l2_reg  # weight decay loss

            # total loss = cross entropy loss + weight decay loss
            loss = cross_entropy_loss + weight_decay * loss_wd

        # backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # update EMA
        ema.update(model)

        # calculate running loss and correct prediction count for accuracy
        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    if scheduler is not None:
        scheduler.step()

    return running_loss / len(train_loader.dataset), correct / len(train_loader.dataset)


# Test function
def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)


def predict(model, loader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            output = nn.Softmax(dim=1)(model(data))
            outputs.append(output)
    return torch.cat(outputs)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(arch, path):
    model = init_network(arch)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
