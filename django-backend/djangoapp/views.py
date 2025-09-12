import json
import threading
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import viewsets, status, permissions, mixins
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework import status
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework.viewsets import GenericViewSet
from django.db.models import Q, Count, F, Case, When, Value, IntegerField
import os
from authlib.integrations.django_client import OAuth
from django.urls import reverse
from .mongodb_utils import get_user_by_email, create_user
from .models import Task, User, Student, Parent, TaskStatusHistory
#from .education_assistant import EducationManager
from .multi_agent_assistant import EducationManager
from .serializers import (
    UserSerializer, StudentSerializer, ParentSerializer,
    TaskSerializer, TaskCreateSerializer, TaskUpdateSerializer,
    TaskSummarySerializer, CustomTokenObtainPairSerializer
)
from .auth_utils import verify_google_token, get_or_create_user_mongodb, get_tokens_for_user

class CustomTokenObtainPairView(TokenObtainPairView):
    """Custom token obtain view that includes user data in the response."""
    serializer_class = CustomTokenObtainPairSerializer

class UserRegistrationView(APIView):
    """View for user registration using MongoDB."""
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        try:
            print("UserRegistrationView post")
            data = json.loads(request.body)
            email = data.get('email')
            password = data.get('password')
            name = data.get('name', '')
            
            if not email or not password:
                return Response(
                    {'error': 'Email and password are required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            import asyncio
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Check if user exists in MongoDB
            print("Check if user exists in MongoDB")
            try:
                # Run the async function and get the result
                existing_user = loop.run_until_complete(
                    get_user_by_email(email)
                )
                if existing_user:
                    return Response(
                        {'error': 'User with this email already exists'},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                # Create new user in MongoDB
                user_data = {
                    'email': email,
                    'password': password,
                    'name': name,
                    'disabled': False
                }
                created_user = loop.run_until_complete(
                    create_user(user_data)
                )
                if not created_user:
                    return Response(
                        {'error': 'Failed to create user'},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                # Return token pair on successful registration
                tokens = get_tokens_for_user(created_user)
                return Response({
                    'message': 'User registered successfully',
                    'user': {
                        'id': created_user['id'],
                        'email': created_user['email'],
                        'name': created_user.get('name', '')
                    },
                    **tokens
                }, status=status.HTTP_201_CREATED)
            except Exception as e:
                print(f"Error checking user registration: {e}")
                return Response(
                    {'error': 'Failed to check user registration'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            finally:
                # Clean up the event loop
                try:
                    loop.close()
                except Exception as e:
                    print(f"Error closing event loop: {str(e)}")
        except json.JSONDecodeError:
            return Response(
                {'error': 'Invalid JSON'},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

# Google OAuth2 config
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")

oauth = OAuth()
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',  # Note: openid-configuration (with hyphen)
    client_kwargs={
        'scope': 'openid email profile'  # You can use short form with Authlib
    }
)
print("GoogleOAuth2 config", GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI)

class GoogleLoginView(APIView):
    """View to initiate Google OAuth login."""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            print("GoogleLoginView - Starting OAuth flow")            
            # Generate redirect URI dynamically
            try:
                redirect_uri = request.build_absolute_uri(reverse('auth_google_callback'))
                redirect_uri = redirect_uri.rstrip('/')
                print(f"GoogleLoginView - Generated redirect_uri: {redirect_uri}")
            except Exception as e:
                print(f"GoogleLoginView - Error generating redirect URI: {str(e)}")
                raise

            # Generate authorization URL
            try:
                print("GoogleLoginView - Creating authorization URL...")
                authorization_url = oauth.google.create_authorization_url(redirect_uri)
                print(f"GoogleLoginView - authorization_url: {authorization_url}")
                
                if not authorization_url or 'url' not in authorization_url:
                    print("GoogleLoginView - Invalid authorization URL response:", authorization_url)
                    raise ValueError("Failed to generate authorization URL")
                    
                if request.GET.get('format') == 'json' or 'application/json' in request.META.get('HTTP_ACCEPT', ''):
                    return Response({
                        'authorization_url': authorization_url['url'],
                        'state': authorization_url.get('state')
                    })
                else:
                    # Redirect directly to Google (easier for browser testing)
                    from django.http import HttpResponseRedirect
                    return HttpResponseRedirect(authorization_url['url'])
                
            except Exception as e:
                print(f"GoogleLoginView - Error creating authorization URL: {str(e)}")
                raise
                
        except Exception as e:
            print(f"GoogleLoginView - Unhandled exception: {str(e)}")
            return Response(
                {'error': 'Failed to initiate Google login', 'details': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class GoogleCallbackView(APIView):
    """Handle Google OAuth callback."""
    permission_classes = [AllowAny]
    
    def dispatch(self, request, *args, **kwargs):
        print(f"GoogleCallbackView.dispatch() called - Method: {request.method}")
        print(f"Full URL: {request.build_absolute_uri()}")
        return super().dispatch(request, *args, **kwargs)
    
    def get(self, request):
        print("GoogleCallbackView.get() called!")
        print(f"Request GET params: {request.GET}")
        print(f"Full path: {request.get_full_path()}")
        
        # Get authorization code from callback
        code = request.GET.get('code')
        state = request.GET.get('state')
        error = request.GET.get('error')
        
        if error:
            print(f"OAuth error: {error}")
            return Response(
                {'error': f'OAuth error: {error}'}, 
                status=400
            )
        
        if not code:
            print("No authorization code provided")
            return Response(
                {'error': 'Authorization code not provided', 'received_params': dict(request.GET)}, 
                status=400
            )
        
        try:
            # Exchange code for token
            redirect_uri = request.build_absolute_uri('/auth/google/callback/').rstrip('/')
            print(f"Using redirect URI: {redirect_uri}")
            
            token = oauth.google.fetch_access_token(
                redirect_uri=redirect_uri,
                code=code
            )
            
            # Get user info
            user_info = oauth.google.get('userinfo', token=token).json()
            print(f"User info received: {user_info.get('email')}")
            
            # Your logic to create/get user and generate tokens
            # user = get_or_create_user(user_info)
            # tokens = get_tokens_for_user(user)
            
            return Response({
                'user_info': user_info,
                'access_token': token.get('access_token'),
                'message': 'Authentication successful'
            })
            
        except Exception as e:
            print(f"Token exchange failed: {e}")
            return Response(
                {'error': f'Authentication failed: {str(e)}'}, 
                status=400
            )

class TaskViewSet(
    mixins.CreateModelMixin,
    mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin,
    mixins.DestroyModelMixin,
    mixins.ListModelMixin,
    GenericViewSet
):
    """
    API endpoint that allows tasks to be viewed, created, updated, or deleted.
    """
    permission_classes = [IsAuthenticated]
    serializer_class = TaskSerializer
    
    def get_queryset(self):
        print(f"get_queryset TaskViewSet")
        # Filter tasks where the user is either the creator or assignee
        return Task.objects.filter(
            Q(created_by=self.request.user) | Q(assigned_to=self.request.user)
        ).order_by('-created_at')
    
    def get_serializer_class(self):
        if self.action == 'create':
            return TaskCreateSerializer
        elif self.action in ['update', 'partial_update']:
            return TaskUpdateSerializer
        return TaskSerializer
    
    @action(detail=True, methods=['post'])
    def complete(self, request, pk=None):
        """Mark a task as completed."""
        task = self.get_object()
        task.completed = True
        task.save()
        
        # Record status change
        TaskStatusHistory.objects.create(
            task=task,
            status=True,
            changed_by=request.user,
            notes="Task marked as completed"
        )
        
        return Response(
            {"status": "Task completed successfully"}, 
            status=status.HTTP_200_OK
        )
    
    @action(detail=True, methods=['post'])
    def uncomplete(self, request, pk=None):
        """Mark a completed task as not completed."""
        task = self.get_object()
        task.completed = False
        task.save()
        
        # Record status change
        TaskStatusHistory.objects.create(
            task=task,
            status=False,
            changed_by=request.user,
            notes="Task marked as not completed"
        )
        
        return Response(
            {"status": "Task uncompleted successfully"}, 
            status=status.HTTP_200_OK
        )

class TaskSummaryView(APIView):
    """API endpoint to get task summary for the current user."""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        print(f"get TaskSummaryView")
        # Get task statistics
        tasks = Task.objects.filter(
            Q(created_by=request.user) | Q(assigned_to=request.user)
        )
        
        total_tasks = tasks.count()
        completed_tasks = tasks.filter(completed=True).count()
        pending_tasks = total_tasks - completed_tasks
        
        completion_rate = (
            (completed_tasks / total_tasks * 100) 
            if total_tasks > 0 else 0
        )
        
        # Get tasks by status
        tasks_by_status = tasks.values('completed').annotate(
            count=Count('id')
        ).order_by('completed')
        
        # Convert to a more readable format
        status_dict = {
            'completed': 0,
            'pending': 0
        }
        
        for item in tasks_by_status:
            if item['completed']:
                status_dict['completed'] = item['count']
            else:
                status_dict['pending'] = item['count']
        
        # Get recent tasks
        recent_tasks = tasks.order_by('-created_at')[:5]
        
        summary = {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'pending_tasks': pending_tasks,
            'completion_rate': round(completion_rate, 2),
            'tasks_by_status': status_dict,
            'recent_tasks': TaskSerializer(recent_tasks, many=True).data
        }
        
        return Response(summary, status=status.HTTP_200_OK)

class StudentTaskViewSet(TaskViewSet):
    """
    API endpoint for student-specific tasks.
    """
    def get_queryset(self):
        print(f"get_queryset StudentTaskViewSet")
        # Only show tasks assigned to the student
        return Task.objects.filter(
            assigned_to=self.request.user,
            task_type='student'
        ).order_by('-created_at')

class ParentTaskViewSet(TaskViewSet):
    """
    API endpoint for parent-specific tasks and assistant functionality.
    """
    _assistant_cache = {}
    _cache_timestamps = {}
    CACHE_TIMEOUT = 3600  # 1 hour
    
    def __init__(self, **kwargs):
        print(f"__init__ ParentTaskViewSet")
        super().__init__(**kwargs)
        self.assistant_manager = EducationManager()
        self.parent_assistant = None

    @classmethod
    def _is_cache_valid(cls, user_id):
        """Check if the cached assistant is still valid."""
        import time
        if user_id not in cls._cache_timestamps:
            return False
        result = (time.time() - cls._cache_timestamps[user_id]) < cls.CACHE_TIMEOUT
        print(f"_is_cache_valid: {result}")
        if (result == False):
            cls._invalidate_user_cache(user_id)
        return result

    @classmethod
    def _invalidate_user_cache(cls, user_id):
        """Invalidate cache for a specific user."""
        if user_id in cls._assistant_cache:
            del cls._assistant_cache[user_id]
        if user_id in cls._cache_timestamps:
            del cls._cache_timestamps[user_id]

    def old_get_parent_assistant(self):
        print(f"self.parent_assistant: {self.parent_assistant}")
        if not self.parent_assistant and hasattr(self, 'request') and hasattr(self.request, 'user'):
            self.parent_assistant = self.assistant_manager.get_assistant('parent', user_id=self.request.user.id)
        return self.parent_assistant
    def get_parent_assistant(self):
        print(f"self.parent_assistant: {self.parent_assistant} {self.request.user.id}")
        if not hasattr(self, 'request') or not hasattr(self.request, 'user'):
            return None
            
        user_id = self.request.user.id
        
        # Check class-level cache
        if user_id in self._assistant_cache and self._is_cache_valid(user_id):
            self.parent_assistant = self._assistant_cache[user_id]
            print(f"Retrieved parent assistant from class cache for user {user_id}")
        else:
            # Create new assistant and cache it
            self.parent_assistant = self.assistant_manager.get_assistant('parent', user_id=user_id)
            
            # Cache at class level
            import time
            self._assistant_cache[user_id] = self.parent_assistant
            self._cache_timestamps[user_id] = time.time()
            print(f"Created and cached new parent assistant for user {user_id}")
        
        return self.parent_assistant

    def create(self, request, *args, **kwargs):
        print(f"create ParentTaskViewSet")
        # Process the message if it exists in the request data
        message = request.data.get('message')
        print(f"Received message: {message}")
        data = request.data.copy()
        print(f"Request data: {data}")
        if message:
            try:
                # Get the parent assistant and process the message
                self.parent_assistant = self.get_parent_assistant()
                if self.parent_assistant:
                    # Run the async process_message in an event loop
                    import asyncio
                    
                    # Create a new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        # Run the async function and get the result
                        result = loop.run_until_complete(
                            self.parent_assistant.process_message(message)
                        )
                        
                        # Handle the response - it should be a tuple of (response, tasks)
                        if isinstance(result, tuple) and len(result) == 2:
                            response, tasks = result
                        else:
                            # Fallback in case the response format is unexpected
                            response = str(result) if result else ""
                            tasks = []
                        
                        # Log the processing
                        print(f"Processed message: {message}")
                        print(f"Assistant response: {response}")
                        print(f"Created tasks: {tasks}")
                        
                        # Add the assistant's response to the request data
                        data['assistant_response'] = str(response)
                        data['created_tasks'] = tasks if isinstance(tasks, list) else []
                    except Exception as e:
                        print(f"Error in process_message: {str(e)}")
                        data['assistant_response'] = "I encountered an error processing your request. Please try again."
                        data['created_tasks'] = []
                    finally:
                        # Clean up the event loop
                        try:
                            loop.close()
                        except Exception as e:
                            print(f"Error closing event loop: {str(e)}")
                
            except Exception as e:
                print(f"Error processing message: {str(e)}")
                # Continue with task creation even if message processing fails
                pass
        
        # Set default values for required fields if not provided
        if 'title' not in data:
            data['title'] = 'New Task'
        if 'description' not in data:
            data['description'] = data.get('message', 'New Task')[:100]
        if 'task_type' not in data:
            data['task_type'] = 'parent'
        if 'assigned_to' not in data and hasattr(request, 'user') and hasattr(request.user, 'email'):
            data['assigned_to'] = request.user.email
            
        # Update request data with the processed data
        request.data.update(data)
        # Ensure the updated data is mutable
        request._full_data = request.data.copy()
        try:
            # Call the parent class's create method to handle the actual task creation
            response = super().create(request, *args, **kwargs)
            
            # If we have an assistant response, include it in the response
            assistant_response = data.get('assistant_response')
            if assistant_response:
                if isinstance(response.data, dict):
                    print("response data is dict")
                    response.data['assistant_response'] = assistant_response
                else:
                    response.data = {
                        **response.data,
                        'assistant_response': assistant_response
                    }
            
            return response
            
        except Exception as e:
            print(f"Error in parent create method: {str(e)}")
            import traceback
            traceback.print_exc()
            return Response(
                {
                    "error": "Failed to create task", 
                    "details": str(e),
                    "assistant_response": data.get('assistant_response', '')
                },
                status=status.HTTP_400_BAD_REQUEST
            )

    def get_queryset(self):
        print(f"get_queryset ParentTaskViewSet")
        # Initialize the parent assistant
        self.parent_assistant = self.get_parent_assistant()
        print(f"self.parent_assistant: {self.parent_assistant}")
        tasks = self.parent_assistant.get_all_tasks()
        print(f"get_queryset tasks: {tasks}")
        return tasks
        """# Get all tasks for the parent's children
        parent = Parent.objects.get(user=self.request.user)
        children = parent.children.all()
        
        return Task.objects.filter(
            assigned_to__in=[child.user for child in children],
            task_type='parent'
        ).order_by('-created_at')"""
    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        return Response({
            "tasks": queryset,
        }, status=status.HTTP_200_OK)

    @action(detail=True, methods=['put', 'delete'], url_path='delete')
    def delete_task(self, request, pk=None):
        """
        Delete a specific task.
        URL: DELETE /parent/tasks/{task_id}/delete/
        """
        print(f"delete_task called for task ID: {pk}")
        try:
            parent_assistant = self.get_parent_assistant()
            if not parent_assistant:
                return Response(
                    {"error": "Parent assistant not available"}, 
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )
            
            # Delete the task
            # Note: You'll need to implement delete_task in MultiAgentEducationAssistant
            success = parent_assistant.delete_task(pk)
            
            if not success:
                return Response(
                    {"error": "Failed to delete task"}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            return Response(
                {"success": True, "message": "Task deleted successfully"},
                status=status.HTTP_204_NO_CONTENT
            )
            
        except Exception as e:
            print(f"Error in delete_task: {str(e)}")
            return Response(
                {
                    "error": "Failed to delete task", 
                    "details": str(e),
                    "task_id": pk
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['put'], url_path='complete')
    def complete_task(self, request, pk=None):
        """
        Complete a specific task.
        URL: PUT /parent/tasks/{task_id}/complete/
        """
        print(f"complete_task called for task ID: {pk}")
        try:
            parent_assistant = self.get_parent_assistant()
            if not parent_assistant:
                return Response(
                    {"error": "Parent assistant not available"}, 
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )
            task = parent_assistant.mark_task_complete(pk)  # Using pk instead of task_id
            if not task:
                return Response(
                    {"error": "Task not found"}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            return Response(
                {"success": True, "message": "Task marked as complete", "task": task},
                status=status.HTTP_200_OK
            )
        except Exception as e:
            print(f"Error in async complete_task: {str(e)}")
            return Response(
                {
                    "error": "Failed to complete task", 
                    "details": str(e),
                    "task_id": pk
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

@csrf_exempt
def index(request):
    """
    The main view that handles user queries.
    """
    if request.method == 'POST':
        # Extract the user's query from the POST data.
        query = request.POST.get('query')
        if not query:
            try:
                body = json.loads(request.body)
                query = body.get('query')
            except (json.JSONDecodeError, AttributeError):
                return JsonResponse(
                    {"error": "Invalid JSON or query parameter"}, 
                    status=400
                )
        
        # Here you would typically process the query and generate a response
        # For now, just return the query as is
        return JsonResponse({"query": query, "response": "This is a placeholder response"})
    
    # For non-POST requests
    return JsonResponse(
        {"message": "Send a POST request with a 'query' parameter"}, 
        status=200
    )

@csrf_exempt
def db_status(request):
    """
    A view to check the status of the database.

    Returns a JSON response indicating whether the database exists and is ready to be queried.
    This is useful for the frontend to decide whether to allow the user to submit queries
    or to show a loading/wait message while the database is being prepared.
    """
    # Check if the database exists using the `database_exists` function from `logic.py`.
    exists = vector_store_exists()
    # Prepare the status message based on the existence of the database.
    status = {
        'exists': exists,
        'message': 'Database exists' if exists else 'Database is being built'
    }
    # Return the status as a JSON response.
    return JsonResponse(status)

@csrf_exempt
def build_db(request):
    """
    A view to initiate the asynchronous building of the database.

    This view starts a new thread to build the database using the `build_database` function
    from `logic.py`, allowing the web server to continue handling other requests.
    This is particularly useful for initial setup or updating the database without downtime.
    """
    thread = threading.Thread(target=build_database)
    thread.daemon = True
    thread.start()
    return JsonResponse({"status": "Database build started in the background"})