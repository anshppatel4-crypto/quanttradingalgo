"""scanner_engine/math_engine.py

GARCH(1,1) engine and helpers to compute True Expected Volatility and compare against market IV.
"""

import numpy as np
import pandas as pd
from arch import arch_model

class GARCHEngine:
    """Fits a GARCH(1,1) on daily log returns and forecasts next-day annualized volatility.

    Usage:
        g = GARCHEngine(prices_series)
        vol = g.forecast_next_day_annual_vol()  # decimal, e.g. 0.25 for 25%
    """

    def __init__(self, prices: pd.Series):
        if prices is None or prices.empty:
            raise ValueError("Prices series is empty")

        # Ensure series sorted by date
        prices = prices.sort_index()
        self.prices = prices
        self.returns = np.log(prices / prices.shift(1)).dropna()
        self.model_result = None

        if len(self.returns) < 30:
            # Too few datapoints to reasonably fit
            self.model_result = None
        else:
            try:
                # Fit on decimal returns
                am = arch_model(self.returns * 1.0, vol='Garch', p=1, q=1, dist='normal', mean='Zero')
                self.model_result = am.fit(disp='off')
            except Exception:
                self.model_result = None

    def forecast_next_day_variance(self):
        """Return next-day variance (decimal^2), or None on failure."""
        if self.model_result is None:
            return None

        try:
            f = self.model_result.forecast(horizon=1, reindex=False)
            var = f.variance.values[-1, 0]
            return float(var)
        except Exception:
            return None

    def forecast_next_day_annual_vol(self):
        """Return next-day annualized volatility (decimal), or None."""
        var_daily = self.forecast_next_day_variance()
        if var_daily is None:
            return None

        # Annualize assuming 252 trading days
        var_annual = var_daily * 252.0
        vol_annual = np.sqrt(var_annual)
        return float(vol_annual)
