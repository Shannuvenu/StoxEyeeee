def check_price_alert(data, threshold_percent=2):
    try:
        if "Close" not in data.columns or len(data["Close"]) < 2:
            return None
        latest = data["Close"].iloc[-1]
        previous = data["Close"].iloc[-2]
        change_percent = ((latest - previous) / previous) * 100
        if abs(change_percent) >= threshold_percent:
            if change_percent > 0:
                return f"📈 Price Alert: UP by {change_percent:.2f}%!"
            else:
                return f"📉 Price Alert: DOWN by {abs(change_percent):.2f}%!"
        return None
    except Exception as e:
        return f"⚠️ Error in price alert: {e}"
def check_volume_alert(data, multiplier=1.5):
    try:
        if "Volume" not in data.columns or len(data["Volume"]) < 2:
            return None
        volume_series = data["Volume"].tail(2)
        avg_volume = volume_series.iloc[0]
        latest_volume = volume_series.iloc[1]
        if latest_volume > avg_volume * multiplier:
            return f"🔔 Volume Alert: Sudden spike detected ({latest_volume:,})"
        return None
    except Exception as e:
        return f"⚠️ Error in volume alert: {e}"
