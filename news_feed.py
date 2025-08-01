import requests
from datetime import datetime, timedelta

NEWSAPI_KEY = "f923f05fbb5543118005533f18a13d5c"
def get_news_feed(symbol):
    try:
        query=symbol.replace(".NS", "")
        today = datetime.now().date()
        last_week = today - timedelta(days=7)
        url=(
            f"https://newsapi.org/v2/everything?"
            f"q={query}&"
            f"from={last_week}&to={today}&"
            f"sortBy=publishedAt&language=en&apiKey={NEWSAPI_KEY}"
        )
        response=requests.get(url)
        if response.status_code==200:
            data=response.json()
            if data.get("articles"):
                return [{"headline": art["title"], "url": art["url"]} for art in data["articles"][:5]]
            else:
                return []
        else:
            return [{"headline": f"News API error: {response.status_code}"}]
    except Exception as e:
        return [{"headline": f"Error fetching news: {e}"}]
