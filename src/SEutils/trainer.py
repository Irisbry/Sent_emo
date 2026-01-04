import torch
import torch.nn as nn
from SEutils.SEmetrics import compute_classification_metrics


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for labels, inputs in loader:
        labels = labels.to(device)
        inputs = inputs.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    for labels, inputs in loader:
        labels = labels.to(device)
        inputs = inputs.to(device)

        logits = model(inputs)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return compute_classification_metrics(
        all_labels,
        all_preds,
        all_probs,
    )
