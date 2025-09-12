"""Test cases for task-related API endpoints."""

from django.urls import reverse
from rest_framework import status

from .base import TestBase


class TaskAPITests(TestBase):
    """Test task-related API endpoints."""

    def test_create_task_as_parent(self):
        """Test creating a task as a parent."""
        # Clear any existing tasks to ensure a clean state
        from ..models import Task
        Task.objects.all().delete()
        
        self.authenticate(self.parent_token)
        url = reverse('task-list')
        data = {
            'title': 'New Math Assignment',
            'description': 'Complete problems 1-20',
            'task_type': 'student',
            'assigned_to': 'student@example.com'
        }
        response = self.client.post(url, data, format='json')
        
        # Debug output to help diagnose the response structure
        print("Response data:", response.data)
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['title'], 'New Math Assignment')
        self.assertEqual(response.data['description'], 'Complete problems 1-20')
        self.assertEqual(response.data['task_type'], 'student')
        
        # Verify the task was created in the database
        task = Task.objects.first()
        self.assertIsNotNone(task, "Task should be created in the database")
        self.assertEqual(task.assigned_to.email, 'student@example.com')
        self.assertEqual(task.created_by.email, 'parent@example.com')
        self.assertFalse(task.completed)

    def test_list_tasks_as_student(self):
        """Test listing tasks as a student."""
        # Clear existing tasks to ensure a clean state
        from ..models import Task
        Task.objects.all().delete()
        
        # Create test tasks specifically for this test
        task1 = Task.objects.create(
            title='Task 1',
            description='Description 1',
            task_type='student',
            created_by=self.parent_user,
            assigned_to=self.student_user,
            completed=False
        )
        task2 = Task.objects.create(
            title='Task 2',
            description='Description 2',
            task_type='student',
            created_by=self.parent_user,
            assigned_to=self.student_user,
            completed=True
        )
        
        # Create a task for a different student (should not appear in results)
        from django.contrib.auth import get_user_model
        User = get_user_model()
        other_student = User.objects.create_user(
            email='other@example.com',
            password='testpass123',
            name='Other Student'
        )
        Task.objects.create(
            title='Other Task',
            description='Should not appear',
            task_type='student',
            created_by=self.parent_user,
            assigned_to=other_student,
            completed=False
        )
        
        # Authenticate and make the request
        self.authenticate(self.student_token)
        url = reverse('student-task-list')
        response = self.client.get(url)
        
        # Debug output to help diagnose the response structure
        print("Response data:", response.data)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Check if the response is paginated (has 'results' key)
        if 'results' in response.data:
            results = response.data['results']
            self.assertEqual(len(results), 2, "Should only see tasks assigned to the authenticated student")
            
            # Verify the tasks are for the authenticated student
            for task in results:
                assigned_to = task.get('assigned_to')
                if isinstance(assigned_to, dict):
                    self.assertEqual(assigned_to.get('email'), 'student@example.com', 
                                   f"Task assigned to wrong student: {assigned_to}")
                else:
                    self.assertEqual(assigned_to, 'student@example.com', 
                                   f"Task assigned to wrong student: {assigned_to}")
        else:
            # If not paginated, just check the direct response
            self.assertEqual(len(response.data), 2, "Should only see tasks assigned to the authenticated student")

    def test_complete_task(self):
        """Test marking a task as complete."""
        # Clear any existing tasks and history
        from ..models import Task, TaskStatusHistory
        TaskStatusHistory.objects.all().delete()
        Task.objects.all().delete()
        
        # Create a new task that's not completed
        task = Task.objects.create(
            id='123e4567-e89b-12d3-a456-426614174000',  # Fixed UUID for testing
            title='Science Homework',
            description='Read chapter 5',
            task_type='student',
            created_by=self.parent_user,
            assigned_to=self.student_user,
            completed=False
        )
        
        # Authenticate as the student
        self.authenticate(self.student_token)
        
        # First, mark the task as complete
        complete_url = reverse('student-task-complete', args=[task.id])
        response = self.client.post(complete_url, {})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data.get('status'), "Task completed successfully", 
                        "Response should indicate task is completed")
        
        # Verify the task was marked as completed in the database
        task.refresh_from_db()
        self.assertTrue(task.completed, "Task should be marked as completed in the database")
        
        # Check that status history was created for the completion
        history = TaskStatusHistory.objects.filter(task=task).order_by('-changed_at').first()
        self.assertIsNotNone(history, "Status history should be created when completing a task")
        self.assertTrue(history.status, "Status should be True for completed task")
        
        # Now, mark the task as incomplete using the uncomplete endpoint
        uncomplete_url = reverse('student-task-uncomplete', args=[task.id])
        response = self.client.post(uncomplete_url, {})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Verify the response indicates the task was marked as not completed
        self.assertEqual(response.data.get('status'), "Task uncompleted successfully", 
                        "Response should indicate task is not completed")
        
        # Verify the task was marked as not completed in the database
        task.refresh_from_db()
        self.assertFalse(task.completed, "Task should be marked as not completed in the database")
        
        # Verify the status history was updated with the uncomplete action
        history_entries = list(TaskStatusHistory.objects.filter(task=task).order_by('changed_at'))
        self.assertEqual(len(history_entries), 2, "Should have two status history entries")
        
        # Check the first history entry (completed)
        self.assertTrue(history_entries[0].status, "First history entry should be completed (True)")
        
        # Check the second history entry (incomplete)
        self.assertFalse(history_entries[1].status, "Second history entry should be incomplete (False)")

    def test_task_summary(self):
        """Test getting task summary."""
        self.authenticate(self.parent_token)
        from ..models import Task
        
        # Get the actual counts from the database
        total_tasks = Task.objects.count()
        completed_tasks = Task.objects.filter(completed=True).count()
        
        url = reverse('task-summary')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['total_tasks'], total_tasks)
        self.assertEqual(response.data['completed_tasks'], completed_tasks)
        self.assertEqual(response.data['pending_tasks'], 1)
        self.assertEqual(response.data['completion_rate'], 50.0)
