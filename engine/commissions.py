"""Correct bilateral commission model.

Fixes the double-count bug in the original backtest/bt_commission.py:
The old code multiplied by 2 *and* set stocklike=True, which caused
Backtrader to charge commission on *both* entry and exit, resulting in
4x the intended commission rate.

This implementation charges per-side at the configured rate.
Backtrader's stocklike=True ensures it fires on both entry and exit,
so we set the per-side rate directly without pre-multiplying.
"""

from __future__ import annotations

import backtrader as bt


class CorrectBilateralCommission(bt.CommInfoBase):
    """Bilateral commission: charges configurable rate on each side (entry + exit).

    For US ETFs:    commission=0.0001 (1 bp per side)
    For A-share ETFs: commission=0.0003 (3 bps per side)
    """

    params = (
        ("commission", 0.0003),       # per-side rate (decimal, e.g. 0.0003 = 3 bps)
        ("min_commission", 5.0),      # minimum commission per trade
        ("stocklike", True),          # charges on both entry AND exit
        ("commtype", None),           # we compute directly in _getcommission
    )

    def _getcommission(self, size: float, price: float, pseudoexec: bool) -> float:
        """Backtrader calls this once per side (entry and exit separately).

        We return the per-side cost directly — stocklike=True ensures
        it's charged on both entry and exit automatically.
        """
        trade_value = abs(size) * price
        commission = trade_value * self.p.commission
        return max(commission, self.p.min_commission)
