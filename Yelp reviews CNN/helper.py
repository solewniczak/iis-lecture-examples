import torch

alphabet='qwertyuiopasdfghjklzxcvbnm1234567890 '


def preprocess_text(text):
    text = text.lower()
    text = [ch for ch in text if ch in alphabet]
    text = ''.join(text)
    return text


def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices)