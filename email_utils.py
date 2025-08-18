import os
from dotenv import load_dotenv
import resend


load_dotenv()
resend.api_key = os.getenv("RESEND_API_KEY")

def send_email_report(to_email, subject, message_body):
    try:
        response = resend.Emails.send({
            "from": "StoxEye <onboarding@resend.dev>", 
            "to": [to_email],
            "subject": subject,
            "html": f"<p>{message_body}</p>"
        })

        print("📩 Email response from Resend:", response)

        return response
    except Exception as e:
        print("❌ Email sending failed:", e)
        return None
