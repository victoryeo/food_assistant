"""Test cases for authentication endpoints."""

from django.urls import reverse
from rest_framework import status

from .base import TestBase


class AuthenticationTests(TestBase):
    """Test authentication endpoints."""

    def test_student_login(self):
        """Test student login with JWT."""
        url = reverse('token_obtain_pair')
        data = {
            'email': 'student@example.com',
            'password': 'testpass123'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access', response.data)
        self.assertIn('refresh', response.data)
        self.assertIn('user', response.data)

    def test_parent_login(self):
        """Test parent login with JWT."""
        url = reverse('token_obtain_pair')
        data = {
            'email': 'parent@example.com',
            'password': 'testpass123'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access', response.data)
        self.assertIn('refresh', response.data)
        self.assertIn('user', response.data)
