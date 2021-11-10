import torch


def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)