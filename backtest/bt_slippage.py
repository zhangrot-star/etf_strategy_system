"""Volume-aware position sizing and slippage configuration for backtrader.

This backtrader fork (1.9.78.123) does not expose SlippageBase.
We use two mechanisms instead:
1. cerebro.broker.set_slippage_perc() for flat proportional slippage
2. VolumeAwareSizer (SizerBase) to limit position size relative to daily volume
"""

from __future__ import annotations

import backtrader as bt


def configure_slippage(cerebro: bt.Cerebro, bps: float = 1.0) -> None:
    """Set proportional slippage on the cerebro broker.

    Args:
        cerebro: The Cerebro instance.
        bps: Slippage in basis points (e.g., 1.0 = 0.01% per trade).
    """
    cerebro.broker.set_slippage_perc(bps / 10000.0)


class VolumeAwareSizer(bt.SizerBase):
    """Position sizer that limits trade size to a fraction of daily volume.

    Prevents large trades from exceeding a configurable percentage of the
    asset's daily volume, which would cause excessive market impact.

    params = (
        ('volume_frac', 0.01),   # max 1% of daily volume per trade
        ('max_frac', 0.30),      # max 30% of portfolio per position
    )
    """

    params = (
        ("volume_frac", 0.01),
        ("max_frac", 0.30),
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        port_value = self.strategy.broker.getvalue()
        max_position_value = port_value * self.p.max_frac
        current_price = data.close[0]

        # Volume cap: don't exceed volume_frac of daily volume
        daily_volume = data.volume[0] if len(data.volume) > 0 else 1e9
        if daily_volume <= 0:
            daily_volume = 1e9
        max_shares_by_volume = daily_volume * self.p.volume_frac

        # Cash-limited shares
        if comminfo:
            max_shares_by_cash = comminfo.getsize(current_price, cash)
        else:
            max_shares_by_cash = int(cash / current_price)

        # Position-limited shares
        max_shares_by_position = int(max_position_value / current_price)

        size = min(max_shares_by_volume, max_shares_by_cash, max_shares_by_position)
        return max(size, 0)
