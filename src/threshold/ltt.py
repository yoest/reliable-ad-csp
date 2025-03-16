import torch
from .risk_function import RiskFunction
from tqdm import tqdm
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix
from .thresholding import Thresholding


class LTT(Thresholding):

    def __init__(self, risk_function: RiskFunction, alpha: float = 0.1, delta: float = 0.1, n_space_lambdas: int = 30, verbose: bool = False, thresholds: tuple = None):
        super().__init__('LTT', thresholds)
        self.risk_function = risk_function
        self.alpha = alpha
        self.delta = delta
        self.n_space_lambdas = n_space_lambdas
        self.verbose = verbose

    def fit(self, calib_scores: torch.tensor, calib_labels: torch.tensor):
        print("Fitting LTT conformalizer", flush=True)  
        self.calib_scores, self.calib_labels = calib_scores, calib_labels
        min_thresh, max_thresh = torch.min(calib_scores), torch.max(calib_scores)  

        # Learn then Test
        lambdas = np.linspace(min_thresh, max_thresh, self.n_space_lambdas)
        p_values = self.compute_p_values(lambdas)
        fwer_lams = self._run_bonferroni(list(p_values.values()), list(p_values.keys()))
        if len(fwer_lams) == 0:
            print("No valid threshold found. Setting thresholds to -inf and inf.")
            self.lam_l, self.lam_h = min_thresh - 100000, max_thresh + 100000
            return
        
        if self.verbose:
            print(f"No. of lambdas: {len(fwer_lams)}")

        # Select the one inducing the min of fpr + fnr + abstention rate)
        best_lam_l, best_lam_h = None, None
        lowest_obj = float('inf')
        pbar = tqdm(fwer_lams, desc=f'Selecting best threshold (Current objective: {lowest_obj:.4f})', disable=not self.verbose)
        for lambda_low, lambda_high in pbar:
            self.lam_l, self.lam_h = lambda_low, lambda_high

            pred_labels = self.classify(calib_scores)
            abstained = pred_labels == -1
            pred_labels_not_abs = pred_labels[~abstained]
            labels_not_abs = calib_labels[~abstained]

            tn, fp, fn, tp = confusion_matrix(labels_not_abs, pred_labels_not_abs, labels=[0, 1]).ravel()
            fpr = fp / (fp + tn) if fp + tn > 0 else 0
            fnr = fn / (fn + tp) if fn + tp > 0 else 0

            obj = fpr + fnr + torch.mean(abstained.float()).item()
            if obj < lowest_obj:
                best_lam_l, best_lam_h = lambda_low, lambda_high
                lowest_obj = obj

                # Update the tqdm description
                pbar.set_description(f'Selecting best threshold (Current objective: {lowest_obj:.4f})')

        self.lam_l, self.lam_h = best_lam_l, best_lam_h

    def compute_p_values(self, lambdas: list) -> dict:
        lambdas_comb = [(ll, lh) for ll in lambdas for lh in lambdas if ll <= lh]

        p_values = {}
        for lambda_low, lambda_high in tqdm(lambdas_comb, desc='Computing p-values', disable=not self.verbose):
            p_values[(lambda_low, lambda_high)] = self._compute_valid_p_value_hoeffding_bentkus(lambda_low, lambda_high, lambdas_comb)
        return p_values

    def _compute_valid_p_value_hoeffding_bentkus(self, cur_lam_l: float, cur_lam_h: float, lambdas_comb: int) -> float:
        self.lam_l, self.lam_h = cur_lam_l, cur_lam_h
        
        n = len(lambdas_comb)
        r_hat = self.get_risk()
        bentkus_p_value = np.e * stats.binom.cdf(np.ceil(n * r_hat), n, self.alpha)
        def h1(a, b):
            with np.errstate(divide='ignore', invalid='ignore'):
                return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
            
        hoeffding_p_value = np.exp(-n * h1(min(r_hat, self.alpha), self.alpha))
        return min(bentkus_p_value, hoeffding_p_value)
        
    def _run_bonferroni(self, p_values: list, lambdas: list) -> list:
        returned_lambdas = []
        for j in range(len(p_values)):
            if p_values[j] <= self.delta / len(lambdas):
                returned_lambdas.append(lambdas[j])
        return returned_lambdas
    
    def get_risk(self):
        pred_fit_labels = self.classify(self.calib_scores)
        fit_labels = self.calib_labels

        # Remove abstained samples
        abstained = pred_fit_labels == -1
        pred_fit_labels = pred_fit_labels[~abstained]
        fit_labels = fit_labels[~abstained]

        return self.risk_function(pred_fit_labels, fit_labels)