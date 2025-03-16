import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix


class Thresholding:

    def __init__(self, method_name: str, thresholds: tuple = None):
        self.method_name = method_name
        if thresholds is not None:
            self.lam_l, self.lam_h = thresholds
        else:
            self.lam_l = None
            self.lam_h = None

    def fit(self, calib_scores: torch.tensor, calib_labels: torch.tensor):
        raise NotImplementedError
    
    def classify(self, scores: torch.tensor):
        # 0 for normal, 1 for anomaly, -1 for abstention
        return (scores <= self.lam_l).int() + (scores >= self.lam_h).int() * 2 - 1
    
    def test(self, test_scores: torch.tensor, test_labels: torch.tensor) -> dict:
        pred_labels = self.classify(test_scores)        

        # Remove abstained samples
        abstained = pred_labels == -1
        pred_labels = pred_labels[~abstained]
        test_labels = test_labels[~abstained]

        # Compute the confusion matrix
        tn, fp, fn, tp = confusion_matrix(test_labels, pred_labels, labels=[0, 1]).ravel()

        fpr = fp / (fp + tn) if fp + tn > 0 else 0
        fnr = fn / (fn + tp) if fn + tp > 0 else 0
        w_fnr_fpr = 0.5 * (fnr + fpr)
        f1 = 2 * (tp) / (2 * tp + fp + fn) if tp + fp + fn > 0 else 1
        abs_rate = torch.mean(abstained.float()).item() if abstained.numel() > 0 else 0

        return {
            'fpr': fpr,
            'fnr': fnr,
            'w_fnr_fpr': w_fnr_fpr,
            'f1': f1,
            'abs_rate': abs_rate
        }
    

class F1ScoreThresholding(Thresholding):

    def __init__(self, thresholds: tuple = None):
        super().__init__('F1-score', thresholds)

    def fit(self, calib_scores: torch.tensor, calib_labels: torch.tensor):
        precision, recall, thresholds_f1 = precision_recall_curve(calib_labels, calib_scores)
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)

        if thresholds_f1.ndim == 0:
            # Special case where recall is 1.0 even for the highest threshold.
            # In this case 'thresholds' will be scalar.
            self.lam_l = thresholds_f1
        else:
            self.lam_l = thresholds_f1[np.argmax(f1_score)]
        self.lam_h = self.lam_l
        

class GMeanThresholding(Thresholding):

    def __init__(self, thresholds: tuple = None):
        super().__init__('G-Mean', thresholds)

    def fit(self, calib_scores: torch.tensor, calib_labels: torch.tensor):
        fpr, tpr, thresholds_gmean = roc_curve(calib_labels, calib_scores)
        gmean = np.sqrt(tpr * (1 - fpr))

        if thresholds_gmean.ndim == 0:
            # Special case where recall is 1.0 even for the highest threshold.
            # In this case 'thresholds' will be scalar.
            self.lam_l = thresholds_gmean
        else:
            self.lam_l = thresholds_gmean[np.argmax(gmean)]
        self.lam_h = self.lam_l


class ZScoreThresholding(Thresholding):

    def __init__(self, z_score_threshold: float = 2.0, thresholds: tuple = None):
        super().__init__('Z-Score', thresholds)
        self.z_score_threshold = z_score_threshold

    def fit(self, calib_scores: torch.tensor, calib_labels: torch.tensor):
        self.lam_l = self.lam_h = self.z_score_threshold

    def classify(self, scores: torch.tensor):
        mu = scores.mean()
        sigma = scores.std()

        z_scores = (scores - mu) / sigma
        z_scores = torch.abs(z_scores)

        return super().classify(z_scores)