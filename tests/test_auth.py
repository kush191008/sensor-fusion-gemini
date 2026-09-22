"""
Unit tests for Production AuthManager and OTP-based Email Verification Engine.
"""

import unittest
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from auth_manager import AuthManager


class TestAuthManager(unittest.TestCase):

    def setUp(self):
        self.auth = AuthManager(otp_validity_seconds=2, max_attempts=5, resend_cooldown_seconds=1)
        self.test_email = "engineer@sensorfusion.ai"

    def test_email_validation(self):
        """Verify RFC compliant email validation."""
        self.assertTrue(AuthManager.is_valid_email("abc@gmail.com"))
        self.assertTrue(AuthManager.is_valid_email("judge@college.edu"))
        self.assertTrue(AuthManager.is_valid_email("user.name+tag@domain.co.uk"))
        self.assertFalse(AuthManager.is_valid_email("invalid-email"))
        self.assertFalse(AuthManager.is_valid_email("@no-user.com"))
        self.assertFalse(AuthManager.is_valid_email(""))

    def test_otp_generation(self):
        """Verify OTP is a 6-digit string and never empty."""
        otp = self.auth.generate_otp(self.test_email)
        self.assertEqual(len(otp), 6)
        self.assertTrue(otp.isdigit())

    def test_otp_verification_success(self):
        """Verify successful OTP verification."""
        otp = self.auth.generate_otp(self.test_email)
        success, msg = self.auth.verify_otp(self.test_email, otp)
        self.assertTrue(success)
        self.assertIn("Email Verified Successfully", msg)
        # Verify OTP is purged after successful verification
        self.assertNotIn(self.test_email, self.auth.active_otps)

    def test_otp_verification_wrong_code(self):
        """Verify wrong OTP code returns incorrect verification message."""
        otp = self.auth.generate_otp(self.test_email)
        wrong_otp = "000000" if otp != "000000" else "111111"
        success, msg = self.auth.verify_otp(self.test_email, wrong_otp)
        self.assertFalse(success)
        self.assertIn("Incorrect Verification Code", msg)

    def test_otp_max_attempts_exceeded(self):
        """Verify account lockout after exceeding 5 maximum attempts."""
        otp = self.auth.generate_otp(self.test_email)
        wrong_otp = "999999"

        # 4 failed attempts
        for _ in range(4):
            self.auth.verify_otp(self.test_email, wrong_otp)

        # 5th failed attempt should trigger lockout
        success, msg = self.auth.verify_otp(self.test_email, wrong_otp)
        self.assertFalse(success)
        self.assertIn("Maximum verification attempts", msg)

        # Subsequent attempts fail because OTP was purged
        success, msg = self.auth.verify_otp(self.test_email, wrong_otp)
        self.assertFalse(success)
        self.assertIn("No active OTP request found", msg)

    def test_otp_expiration(self):
        """Verify OTP expiration after 5 minutes (simulated with 2s window)."""
        otp = self.auth.generate_otp(self.test_email)
        time.sleep(2.2)
        success, msg = self.auth.verify_otp(self.test_email, otp)
        self.assertFalse(success)
        self.assertIn("expired", msg.lower())

    def test_resend_cooldown(self):
        """Verify resend cooldown mechanism."""
        self.auth.generate_otp(self.test_email)
        allowed, rem = self.auth.can_resend(self.test_email)
        self.assertFalse(allowed)
        self.assertGreaterEqual(rem, 1)

        time.sleep(1.2)
        allowed, rem = self.auth.can_resend(self.test_email)
        self.assertTrue(allowed)

    def test_session_management(self):
        """Verify login and logout session state helpers."""
        session = {}
        self.auth.login_session(session, self.test_email)
        self.assertTrue(session["authenticated"])
        self.assertEqual(session["user_email"], self.test_email)
        self.assertIn("session_id", session)

        self.auth.logout_session(session)
        self.assertFalse(session.get("authenticated", False))


if __name__ == '__main__':
    unittest.main()
