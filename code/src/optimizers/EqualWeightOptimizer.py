from .BaseOptimizer import BaseOptimizer
import numpy as np

class EqualWeightOptimizer(BaseOptimizer):
    """
    1/N benchmark portfolio.
    Allocates wealth equally across all available assets and 
    assumes zero active currency overlay (fully hedged baseline).
    """
    
    def __init__(self, hyperparams=None):
        super().__init__(hyperparams)
        self.num_assets = 0

    def fit(self, train_bundle):
        """
        The 1/N portfolio doesn't need historical returns to train.
        It only needs to know how many assets (N) exist in the universe.
        """
        R_fh = train_bundle.get('R_fh')
        if R_fh is None:
            raise ValueError("train_bundle must contain 'R_fh' (asset returns).")

        self.num_assets = R_fh.shape[1]

    def optimize(self):
        """
        Returns a vector where every asset gets exactly 1/N of the capital.
        """
        if self.num_assets == 0:
            raise ValueError("Optimizer must be fitted before calling optimize().")

        return np.full(self.num_assets, 1.0 / self.num_assets)
        
    def score(self, weights, val_bundle):
        """
        Because 1/N has no hyperparameters.
        """
        return 0.0
