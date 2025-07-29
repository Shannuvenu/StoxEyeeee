def check_price_alert(data,threshold_percent=2):
    try:
        latest=data["Close"].iloc[-1]
        previous=data["Close"].iloc[-2]
        change_percent=((latest-previous)/previous)*100
        if abs(change-change_percent)>=threshold_percent:
            if change_percent>0:
                return f"price alert: up by change{change_percent:.2f}%!"
            else:
                return f"price alert: down by {abs(change_percent):.2f}%!"
            return None
    except Exception as e:
        return f"error in price alert:{e}"
def check_volume_alert(data,multiplier=1.5):
        try:
            volume=data["Volume"].tail(2)
            avg_volume=volume[:-1].mean()
            latest_volume=volumes.iloc[-1]

            if latest_volume>avg_volume*multiplier:
               return f"volume alert: sudden spike detected ({latest_volume:,})"
            return None
        except Exception as e:
            return f"error in volume alert:{e}"

            