import os
import resend
from dotenv import load_dotenv
load_dotenv()
resend.api_key = os.getenv("RESEND_API_KEY")

def send_email_report(to_email, subject, html_content):
    try:
        params = {
            "from": "StoxEye <onboarding@resend.dev>",
            "to": [to_email],
            "subject": subject,
            "html": html_content
        }
        result = resend.Emails.send(params)
        return True if result.get("id") else False
    except Exception as e:
        return f"Error: {e}"
