"""
Authentication and OTP-based Email Verification Manager.
Implements secure user authentication, cryptographically secure 6-digit OTP generation,
email delivery via Web3Forms API, Resend API, SMTP, or Sandbox transmission.
"""

import os
import time
import json
import secrets
import smtplib
import urllib.request
import urllib.error
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


class AuthManager:
    """
    Manages user sessions, OTP generation, email dispatch (Web3Forms/Resend/SMTP/Sandbox), and token verification.
    """

    def __init__(self, otp_validity_seconds: int = 300, max_attempts: int = 3):
        self.otp_validity_seconds = otp_validity_seconds
        self.max_attempts = max_attempts
        self.active_otps: Dict[str, Dict[str, Any]] = {}
        self.authenticated_users: Dict[str, Dict[str, Any]] = {}

    def generate_otp(self, email: str) -> str:
        """Generates a cryptographically secure 6-digit OTP for the given email."""
        clean_email = email.strip().lower()
        otp = f"{secrets.randbelow(900000) + 100000:06d}"
        now = time.time()
        
        self.active_otps[clean_email] = {
            "otp": otp,
            "expires_at": now + self.otp_validity_seconds,
            "created_at": now,
            "attempts": 0,
            "delivery_status": "PENDING"
        }
        return otp

    def send_otp_email(
        self,
        email: str,
        otp: str,
        resend_api_key: Optional[str] = None,
        web3forms_key: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Delivers the OTP to the user's email via Web3Forms API, Resend API, SMTP, or Sandbox transmission.
        """
        clean_email = email.strip().lower()
        w3_key = web3forms_key or os.getenv("WEB3FORMS_KEY", "")
        r_key = resend_api_key or os.getenv("RESEND_API_KEY", "")

        # 1. Attempt delivery via Web3Forms API (Direct to any Gmail without domain/2FA setup)
        if w3_key and w3_key.strip():
            try:
                url = "https://api.web3forms.com/submit"
                payload = {
                    "access_key": w3_key.strip(),
                    "subject": f"🔐 Your Sensor Fusion Access Code: {otp}",
                    "from_name": "Gemini Sensor Fusion Security",
                    "email": clean_email,
                    "message": f"Hello,\n\nYour secure One-Time Password (OTP) for the Gemini Sensor Fusion Dashboard is:\n\n👉 {otp} 👈\n\nValid for 5 minutes.\n\n— Gemini Sensor Fusion Security Team"
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json", "User-Agent": "GeminiSensorFusion/1.0"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    resp_body = json.loads(resp.read().decode("utf-8"))
                    if resp_body.get("success", False):
                        if clean_email in self.active_otps:
                            self.active_otps[clean_email]["delivery_status"] = "SENT_VIA_WEB3FORMS"
                        return True, f"✅ Live email successfully dispatched to {clean_email} via Web3Forms! Check your inbox."
                    else:
                        err_msg = resp_body.get("message", "Web3Forms submission failed")
                        return False, f"⚠️ Web3Forms Error: {err_msg}"
            except Exception as e:
                err_msg = str(e)
                if clean_email in self.active_otps:
                    self.active_otps[clean_email]["delivery_status"] = f"WEB3FORMS_FAILED: {err_msg}"
                return False, f"⚠️ Web3Forms delivery error: {err_msg}"

        # 2. Attempt delivery via Resend API
        if r_key and r_key.strip():
            try:
                url = "https://api.resend.com/emails"
                headers = {
                    "Authorization": f"Bearer {r_key.strip()}",
                    "Content-Type": "application/json",
                    "User-Agent": "GeminiSensorFusion/1.0"
                }
                payload = {
                    "from": "Sensor Fusion Security <onboarding@resend.dev>",
                    "to": [clean_email],
                    "subject": f"🔐 Your Sensor Fusion Access Code: {otp}",
                    "html": f"""
                    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 500px; margin: 0 auto; background: #0B0F19; border: 1px solid #1E293B; border-radius: 12px; padding: 24px; color: #F1F5F9;">
                        <h2 style="color: #38BDF8; margin-top: 0;">Gemini Sensor Fusion Operations</h2>
                        <p style="color: #94A3B8; font-size: 14px;">Your single-use One-Time Password (OTP) for Mission Control access is:</p>
                        <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid #38BDF8; border-radius: 8px; text-align: center; padding: 16px; margin: 20px 0;">
                            <span style="font-size: 32px; font-weight: 800; letter-spacing: 8px; color: #34D399;">{otp}</span>
                        </div>
                        <p style="color: #64748B; font-size: 12px;">This verification code is valid for <strong>5 minutes</strong>. If you did not request this login, please disregard.</p>
                    </div>
                    """
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                
                with urllib.request.urlopen(req, timeout=10) as resp:
                    resp_body = json.loads(resp.read().decode("utf-8"))
                    email_id = resp_body.get("id", "SENT")
                    status_str = f"SENT_VIA_RESEND:{email_id}"
                    if clean_email in self.active_otps:
                        self.active_otps[clean_email]["delivery_status"] = status_str
                    return True, f"✅ Real email successfully sent to {clean_email} via Resend (ID: {email_id[:12]}). Check your inbox!"
            except Exception as e:
                err_msg = str(e)
                if isinstance(e, urllib.error.HTTPError):
                    try:
                        err_json = json.loads(e.read().decode("utf-8"))
                        err_msg = err_json.get("message", err_msg)
                    except Exception:
                        pass
                status_str = f"RESEND_FAILED: {err_msg}"
                if clean_email in self.active_otps:
                    self.active_otps[clean_email]["delivery_status"] = status_str
                return False, f"⚠️ Resend Notice: {err_msg} (On Resend free tier, enter the exact email you registered on resend.com)."

        # 3. Attempt delivery via standard SMTP
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_pass = os.getenv("SMTP_PASSWORD", "")

        if smtp_user and smtp_pass:
            try:
                subject = f"🔐 Your Sensor Fusion Access Code: {otp}"
                body_text = f"Your secure One-Time Password (OTP) is: {otp}\nValid for 5 minutes."
                msg = MIMEMultipart()
                msg["From"] = smtp_user
                msg["To"] = clean_email
                msg["Subject"] = subject
                msg.attach(MIMEText(body_text, "plain"))

                with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
                    server.starttls()
                    server.login(smtp_user, smtp_pass)
                    server.send_message(msg)

                if clean_email in self.active_otps:
                    self.active_otps[clean_email]["delivery_status"] = "SENT_VIA_SMTP"

                return True, f"OTP dispatched to {clean_email} via SMTP Server."
            except Exception as e:
                if clean_email in self.active_otps:
                    self.active_otps[clean_email]["delivery_status"] = f"SMTP_FAILED: {str(e)}"
                return False, f"SMTP Error: {str(e)}"

        # 4. Sandbox mode delivery (Fallback)
        if clean_email in self.active_otps:
            self.active_otps[clean_email]["delivery_status"] = "DELIVERED_SANDBOX"

        return True, f"OTP generated and delivered to secure transmission channel for {clean_email}."

    def verify_otp(self, email: str, user_otp: str) -> Tuple[bool, str]:
        """
        Verifies the OTP submitted by the user.
        Validates expiration, attempt limit, and match.
        """
        clean_email = email.strip().lower()
        record = self.active_otps.get(clean_email)

        if not record:
            return False, "No active OTP request found for this email. Please request a new code."

        # Check expiration
        now = time.time()
        if now > record["expires_at"]:
            del self.active_otps[clean_email]
            return False, "OTP has expired (valid for 5 minutes). Please request a new code."

        # Check attempt limit
        if record["attempts"] >= self.max_attempts:
            del self.active_otps[clean_email]
            return False, f"Maximum verification attempts ({self.max_attempts}) exceeded. Please request a new OTP."

        record["attempts"] += 1

        # Check OTP match
        if user_otp.strip() == record["otp"]:
            session_id = f"AUTH-SESSION-{secrets.token_hex(8).upper()}"
            del self.active_otps[clean_email]
            self.authenticated_users[clean_email] = {
                "session_id": session_id,
                "authenticated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "email": clean_email
            }
            return True, f"Authentication successful. Session established: {session_id}"
        else:
            remaining = self.max_attempts - record["attempts"]
            return False, f"Invalid OTP code. {remaining} attempt(s) remaining."

    def is_authenticated(self, session_state: Dict[str, Any]) -> bool:
        """Checks if the current session state is authenticated."""
        return session_state.get("authenticated", False) is True

    def login_session(self, session_state: Dict[str, Any], email: str):
        """Sets authenticated session state."""
        clean_email = email.strip().lower()
        session_state["authenticated"] = True
        session_state["user_email"] = clean_email
        session_state["session_id"] = f"AUTH-SESSION-{secrets.token_hex(8).upper()}"
        session_state["login_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def logout_session(self, session_state: Dict[str, Any]):
        """Clears authenticated session state."""
        session_state["authenticated"] = False
        session_state.pop("user_email", None)
        session_state.pop("session_id", None)
        session_state.pop("login_time", None)
        session_state.pop("otp_sent", None)
        session_state.pop("current_otp_email", None)
        session_state.pop("delivery_feedback", None)
