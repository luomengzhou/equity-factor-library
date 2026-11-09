"""
Market regime detection + regime-conditional factor validation.

Three regime dimensions:
  trend / chop     — rolling Sharpe-like score on the market return
  bull / bear      — rolling cumulative market return with expanding quantile
  high_vol / low_vol — rolling market return std with expanding quantile

The analysis layer then computes grouped IC (factor × regime).
"""
from factor_library.regime.detection import (
    RegimeConfig,
    VolBullBearConfig,
    compute_market_return_from_prices,
    label_regime_from_market_return,
    label_volatility_regime,
    label_bull_bear_regime,
)
from factor_library.regime.validation import (
    compute_ic_summary_vs_forward_return_grouped,
    summarize_ic_series,
)

__all__ = [
    "RegimeConfig", "VolBullBearConfig",
    "compute_market_return_from_prices",
    "label_regime_from_market_return",
    "label_volatility_regime",
    "label_bull_bear_regime",
    "compute_ic_summary_vs_forward_return_grouped",
    "summarize_ic_series",
]
