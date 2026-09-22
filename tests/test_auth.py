"""
Unit tests for AuthManager and OTP-based Email Verification (Challenge 02).
"""

import unittest
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from auth_manager import AuthManager


class TestAuthManager(unittest.TestCase):

    def setUp(self):
        self.auth = AuthManager(otp_validity_seconds=2, max_attempts=3)
        self.test_email = "engineer@sensorfusion.ai"

    def test_otp_generation(self):
        """Verify OTP is a 6-digit string."""
        otp = self.auth.generate_otp(self.test_email)
        self.assertEqual(len(otp), 6)
        self.assertTrue(otp.isdigit())

    def test_otp_verification_success(self):
        """Verify successful OTP verification."""
        otp = self.auth.generate_otp(self.test_email)
        success, msg = self.auth.verify_otp(self.test_email, otp)
        self.assertTrue(success)
        self.assertIn("Authentication successful", msg)

    def test_otp_verification_wrong_code(self):
        """Verify wrong OTP code reduces remaining attempts."""
        otp = self.auth.generate_otp(self.test_email)
        wrong_otp = "000000" if otp != "000000" else "111111"
        success, msg = self.auth.verify_otp(self.test_email, wrong_otp)
        self.assertFalse(success)
        self.assertIn("Invalid OTP", msg)

    def test_otp_max_attempts_exceeded(self):
        """Verify account lockout after exceeding maximum attempts."""
        otp = self.auth.generate_otp(self.test_email)
        wrong_otp = "999999"

        # Attempt 1 (attempts becomes 1)
        self.auth.verify_otp(self.test_email, wrong_otp)
        # Attempt 2 (attempts becomes 2)
        self.auth.verify_otp(self.test_email, wrong_otp)
        # Attempt 3 (attempts becomes 3)
        self.auth.verify_otp(self.test_email, wrong_otp)

        # Attempt 4 should fail due to exceeding max attempts
        success, msg = self.auth.verify_otp(self.test_email, wrong_otp)
        self.assertFalse(success)
        self.assertIn("Maximum verification attempts", msg)

        # Attempt 5 should fail because OTP was invalidated
        success, msg = self.auth.verify_otp(self.test_email, wrong_otp)
        self.assertFalse(success)
        self.assertIn("No active OTP request found", msg)

    def test_otp_expiration(self):
        """Verify OTP expiration after validity window."""
        otp = self.auth.generate_otp(self.test_email)
        # Wait for expiration (validity was set to 2 seconds in setUp)
        time.sleep(2.5)
        success, msg = self.auth.verify_otp(self.test_email, otp)
        self.assertFalse(success)
        self.assertIn("expired", msg.lower())

    def test_session_management(self):
        """Verify login and logout session state helpers."""
        session = {}
        self.assertFalse(self.auth.is_authenticated(session))
        
        self.auth.login_session(session, self.test_email)
        self.assertTrue(self.auth.is_authenticated(session))
        self.assertEqual(session["user_email"], self.test_email)
        self.assertIn("session_id", session)

        self.auth.logout_session(session)
        self.assertFalse(self.auth.is_authenticated(session))


if __name__ == '__main__':
    unittest.main()
