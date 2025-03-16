import numpy as np
from sklearn.metrics import confusion_matrix
import torch


class RiskFunction:

    def __init__(self, risk_name: str, B: int = 1):
        self.risk_name = risk_name
        self.B = B

    def __call__(self, pred_labels: np.array, labels: np.array):
        raise NotImplementedError
    

class FNRFunction(RiskFunction):

    def __init__(self):
        super().__init__('FNR')

    def __call__(self, pred_labels: torch.Tensor, labels: torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
        labels = labels.cpu().numpy()

        abstained = pred_labels == -1
        pred_labels_not_abs = pred_labels[~abstained]
        labels_not_abs = labels[~abstained]

        tn, fp, fn, tp = confusion_matrix(labels_not_abs, pred_labels_not_abs, labels=[0, 1]).ravel()
        fnr = fn / (fn + tp) if fn + tp > 0 else 0
        return fnr
    

class FPRFunction(RiskFunction):

    def __init__(self):
        super().__init__('FPR')

    def __call__(self, pred_labels: torch.Tensor, labels: torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
        labels = labels.cpu().numpy()

        tn, fp, fn, tp = confusion_matrix(labels, pred_labels, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if fp + tn > 0 else 0
        return fpr
    

class WeightedFPRFNRFunction(RiskFunction):

    def __init__(self):
        super().__init__('WeightedFPRFNR')

    def __call__(self, pred_labels: torch.Tensor, labels: torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
        labels = labels.cpu().numpy()

        abstained = pred_labels == -1
        pred_labels_not_abs = pred_labels[~abstained]
        labels_not_abs = labels[~abstained]

        tn, fp, fn, tp = confusion_matrix(labels_not_abs, pred_labels_not_abs, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if fp + tn > 0 else 0
        fnr = fn / (fn + tp) if fn + tp > 0 else 0

        return 0.5 * fpr + 0.5 * fnr
    

class F1Function(RiskFunction):

    def __init__(self):
        super().__init__('F1')

    def __call__(self, pred_labels: torch.Tensor, labels: torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
        labels = labels.cpu().numpy()

        abstained = pred_labels == -1
        pred_labels_not_abs = pred_labels[~abstained]
        labels_not_abs = labels[~abstained]

        tn, fp, fn, tp = confusion_matrix(labels_not_abs, pred_labels_not_abs, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if tp + fp > 0 else 1
        recall = tp / (tp + fn) if tp + fn > 0 else 1

        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        # A higher F1 score is better, so we return 1 - F1
        return 1 - f1