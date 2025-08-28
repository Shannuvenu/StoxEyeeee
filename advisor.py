# advisor.py
from typing import Dict, Tuple
import numpy as np
import pandas as pd

def analyze_stock(data: pd.DataFrame, symbol: str, alerts: Dict[str, str], vol: float) -> Tuple[str, str]:
    """
    Returns (signal, reason) where signal ∈ {"BUY","SELL","HOLD"}.
    Uses basic trend + volume + volatility + alerts.
    """
    if data is None or data.empty or "Close" not in data.columns:
        return "HOLD", "❓ No data available"

    reason_parts = []

    # Recent price trend
    recent = data["Close"].tail(20).pct_change().mean() * 100
    if recent > 1:
        reason_parts.append(f"Uptrend {recent:.1f}% over last 20 sessions")
    elif recent < -1:
        reason_parts.append(f"Downtrend {recent:.1f}% over last 20 sessions")

    # Alerts
    if "price" in alerts and alerts["price"]:
        reason_parts.append(f"Price Alert: {alerts['price']}")
    if "volume" in alerts and alerts["volume"]:
        reason_parts.append(f"Volume Alert: {alerts['volume']}")

    # Volatility
    if np.isfinite(vol):
        if vol >= 35:
            reason_parts.append(f"High volatility {vol:.1f}% annualized")
        elif vol < 20:
            reason_parts.append(f"Stable low volatility {vol:.1f}%")

    # Final Signal Logic
    signal = "HOLD"
    if recent > 1.5 and vol < 35:
        signal = "BUY"
    elif recent < -1.5 or "Downtrend" in "".join(reason_parts):
        signal = "SELL"

    # Combine reasons
    if not reason_parts:
        reason_parts.append("Neutral signals. Hold for now.")

    return signal, " | ".join(reason_parts)
