import numpy as np
from dataclasses import dataclass

@dataclass
class RiskBudget:
    gross_leverage: float
    max_single_stock: float
    sector_limit: float
    cash_buffer: float

class RiskBudgetManager:
    def __init__(self, min_kappa: float = 0.20, max_kappa: float = 1.00):
        self.min_kappa = min_kappa
        self.max_kappa = max_kappa

    def compute_budget(self, kappa: float) -> RiskBudget:
        kappa_clamped = float(np.clip(kappa, self.min_kappa, self.max_kappa))
        gross_leverage = kappa_clamped
        max_single_stock = float(np.clip(0.05 * kappa_clamped, 0.02, 0.05))
        sector_limit = float(np.clip(0.25 * kappa_clamped, 0.12, 0.25))
        cash_buffer = float(1.0 - kappa_clamped)
        
        return RiskBudget(
            gross_leverage=gross_leverage,
            max_single_stock=max_single_stock,
            sector_limit=sector_limit,
            cash_buffer=cash_buffer
        )
