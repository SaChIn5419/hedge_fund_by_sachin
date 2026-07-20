from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseAlphaEngine(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def expected_returns(self, current_date: pd.Timestamp) -> pd.Series:
        """Returns cross-sectional expected returns for universe at date."""
        pass

    @abstractmethod
    def confidence_weights(self, current_date: pd.Timestamp) -> pd.Series:
        """Returns signal confidence weights [0, 1] per ticker."""
        pass

    @abstractmethod
    def covariance_adjustment(self, current_date: pd.Timestamp, base_cov: pd.DataFrame) -> pd.DataFrame:
        """Returns adjusted covariance matrix for date."""
        pass

class FileBasedAlphaEngine(BaseAlphaEngine):
    """Alpha engine wrapper reading pre-computed weekly strategy returns."""
    def __init__(self, name: str, returns_filepath: str):
        super().__init__(name)
        self.df_returns = pd.read_csv(returns_filepath, parse_dates=['date']).set_index('date').sort_index()

    def expected_returns(self, current_date: pd.Timestamp) -> pd.Series:
        if current_date in self.df_returns.index:
            ret_val = self.df_returns.loc[current_date, 'portfolio_return']
            return pd.Series({'PORTFOLIO': ret_val})
        return pd.Series({'PORTFOLIO': 0.0})

    def confidence_weights(self, current_date: pd.Timestamp) -> pd.Series:
        return pd.Series({'PORTFOLIO': 1.0})

    def covariance_adjustment(self, current_date: pd.Timestamp, base_cov: pd.DataFrame) -> pd.DataFrame:
        return base_cov
