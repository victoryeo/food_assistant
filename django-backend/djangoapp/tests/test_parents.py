"""Test cases for parent-specific API endpoints."""

from django.urls import reverse
from rest_framework import status

from .base import TestBase


class ParentAPITests(TestBase):
    """Test parent-specific API endpoints."""

    def setUp(self):
        """Set up test data for parent tests."""
        super().setUp()
        # Clear any existing tasks
        from ..models import Task
        Task.objects.all().delete()
        
        # Create tasks for the parent's child with task_type='parent'
        self.task1 = Task.objects.create(
            title='Math Homework',
            description='Complete exercises 1-10',
            task_type='parent',
            created_by=self.parent_user,
            assigned_to=self.student_user,
            completed=False
        )
        self.task2 = Task.objects.create(
            title='Science Project',
            description='Work on the science project',
            task_type='parent',
            created_by=self.parent_user,
            assigned_to=self.student_user,
            completed=True
        )

    def test_parent_can_see_child_tasks(self):
        """Test that a parent can see their child's tasks."""
        self.authenticate(self.parent_token)
        url = reverse('parent-task-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Get the response data, handling pagination if present
        response_data = response.data
        if 'results' in response_data:  # Handle paginated responses
            response_data = response_data['results']
            
        # Should see both parent tasks for their child
        self.assertEqual(len(response_data), 2, f"Expected 2 tasks, got {len(response_data)}. Response data: {response_data}")
        
        # Verify the tasks are for the parent's child and have task_type='parent'
        for task in response_data:
            self.assertEqual(task['assigned_to']['email'], 'student@example.com')
            self.assertEqual(task['task_type'], 'parent')
    
    def test_create_task_as_parent(self):
        """Test that a parent can create a task for their child."""
        from datetime import datetime
        from django.utils import timezone
        from ..models import Task, User
        
        # Clear any existing tasks to ensure a clean state
        Task.objects.all().delete()
        
        self.authenticate(self.parent_token)
        url = reverse('parent-task-list')
        
        # Create a task for the parent's child
        due_date = timezone.now() + timezone.timedelta(days=7)
        task_data = {
            'title': 'New Parent Task',
            'description': 'This is a test task created by parent',
            'assigned_to': 'student@example.com',  # Parent's child
            'task_type': 'parent',  # Required field
            'due_date': due_date.isoformat(),
            'completed': False
        }
        
        response = self.client.post(url, task_data, format='json')
        
        # Check if we got a 201 Created response
        self.assertEqual(response.status_code, status.HTTP_201_CREATED, 
                        f"Expected status code 201, got {response.status_code}. Response: {response.data}")
        
        # Verify the task was created in the database with the correct attributes
        tasks = Task.objects.all()
        self.assertEqual(tasks.count(), 1, "Expected exactly one task to be created")
        
        task = tasks.first()
        self.assertEqual(task.title, task_data['title'])
        self.assertEqual(task.description, task_data['description'])
        
        # Verify assigned_to is set correctly to the student
        self.assertEqual(task.assigned_to.email, task_data['assigned_to'])
        
        # Verify other task properties
        self.assertEqual(task.task_type, 'parent')
        self.assertEqual(task.created_by, self.parent_user)
        self.assertFalse(task.completed)
        self.assertIsNotNone(task.due_date)
        
        # Verify the response contains the expected fields
        response_data = response.data
        expected_fields = ['title', 'description', 'task_type', 'due_date', 'parent_task']
        for field in expected_fields:
            self.assertIn(field, response_data, f"Expected field '{field}' in response")
        
        # Verify the response data matches what we sent
        self.assertEqual(response_data['title'], task_data['title'])
        self.assertEqual(response_data['description'], task_data['description'])
        self.assertEqual(response_data['task_type'], 'parent')
        
        # Handle potential timezone format differences (Z vs +00:00)
        response_due_date = response_data['due_date'].replace('Z', '+00:00')
        expected_due_date = task_data['due_date'].replace('Z', '+00:00')
        self.assertEqual(response_due_date, expected_due_date)
        
        self.assertIsNone(response_data['parent_task'])
    
    def test_parent_cannot_see_other_children_tasks(self):
        """Test that a parent can only see their own children's tasks."""
        # Create another student not related to the parent
        from django.contrib.auth import get_user_model
        from ..models import Student, Task
        
        User = get_user_model()
        other_student_user = User.objects.create_user(
            email='other@example.com',
            password='testpass123',
            name='Other Student'
        )
        other_student = Student.objects.create(
            user=other_student_user,
            grade_level='9',
            date_of_birth='2011-01-01'
        )
        
        # Create a task for the other student
        Task.objects.create(
            title='Private Task',
            description='Should not be visible',
            task_type='parent',
            created_by=other_student_user,
            assigned_to=other_student_user,
            completed=False
        )
        
        # Authenticate as the original parent
        self.authenticate(self.parent_token)
        url = reverse('parent-task-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Get the response data, handling pagination if present
        response_data = response.data
        if 'results' in response_data:  # Handle paginated responses
            response_data = response_data['results']
        
        # Should only see tasks for their own children (2 parent tasks we created in setUp)
        self.assertEqual(len(response_data), 2, f"Expected 2 tasks, got {len(response_data)}. Response data: {response_data}")
        
        # Verify the tasks are for the parent's child and have task_type='parent'
        for task in response_data:
            self.assertEqual(task['assigned_to']['email'], 'student@example.com')
            self.assertEqual(task['task_type'], 'parent')
