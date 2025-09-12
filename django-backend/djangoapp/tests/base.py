"""Base test class with common setup for all test cases."""

from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.test import APITestCase, APIClient
from rest_framework_simplejwt.tokens import RefreshToken

from ..models import Task, Student, Parent

User = get_user_model()


class TestBase(APITestCase):
    """Base test class with common setup for all test cases."""

    def setUp(self):
        """Set up test data for all test cases."""
        self.client = APIClient()
        
        # Create test users
        self.student_user = User.objects.create_user(
            email='student@example.com',
            password='testpass123',
            name='Test Student'
        )
        
        self.parent_user = User.objects.create_user(
            email='parent@example.com',
            password='testpass123',
            name='Test Parent'
        )
        
        # Create student and parent profiles
        self.student = Student.objects.create(
            user=self.student_user,
            grade_level='10',
            date_of_birth='2010-01-01'
        )
        
        self.parent = Parent.objects.create(user=self.parent_user)
        self.parent.children.add(self.student)
        
        # Create some test tasks
        self.task1 = Task.objects.create(
            title='Math Homework',
            description='Complete exercises 1-10',
            task_type='student',
            created_by=self.parent_user,
            assigned_to=self.student_user
        )
        
        self.task2 = Task.objects.create(
            title='Science Project',
            description='Work on the volcano project',
            task_type='student',
            created_by=self.parent_user,
            assigned_to=self.student_user,
            completed=True
        )
        
        # Get JWT tokens
        self.student_token = self.get_tokens_for_user(self.student_user)
        self.parent_token = self.get_tokens_for_user(self.parent_user)
    
    def get_tokens_for_user(self, user):
        """Generate JWT tokens for a user.
        
        Args:
            user: User instance to generate tokens for
            
        Returns:
            str: Access token for the user
        """
        refresh = RefreshToken.for_user(user)
        return str(refresh.access_token)
    
    def authenticate(self, token):
        """Set up authentication for test client.
        
        Args:
            token (str): JWT token for authentication
        """
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')
