from django.urls import path, include
from rest_framework import routers
from rest_framework_simplejwt.views import (
    TokenRefreshView,
    TokenVerifyView,
)
from . import views
from django.http import HttpResponse

# Create a router for our API endpoints
router = routers.DefaultRouter()
router.register(r'tasks', views.TaskViewSet, basename='task')
router.register(r'student/tasks', views.StudentTaskViewSet, basename='student-tasks')
router.register(r'parent/tasks', views.ParentTaskViewSet, basename='parent-tasks')

# API URL patterns
api_patterns = [
    # Authentication
    path('register/', views.UserRegistrationView.as_view(), name='register'),
    path('token', views.CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/', views.CustomTokenObtainPairView.as_view(), name='token_obtain_pair_slash'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('token/verify/', TokenVerifyView.as_view(), name='token_verify'),
    path('auth/google/', views.GoogleLoginView.as_view(), name='auth_google'),
    path('auth/google/callback', views.GoogleCallbackView.as_view(), name='auth_google_callback'),
    # Task management
    path('tasks/summary/', views.TaskSummaryView.as_view(), name='task-summary'),
    
    # Explicit route for complete task action (handles both with and without trailing slash)
    path('parent/tasks/<uuid:pk>/complete/', 
         views.ParentTaskViewSet.as_view({'put': 'complete_task'}), 
         name='parent-task-complete'),
    path('parent/tasks/<uuid:pk>/complete',  # Without trailing slash
         views.ParentTaskViewSet.as_view({'put': 'complete_task'}), 
         name='parent-task-complete-noslash'),
    # Explicit route for delete task action (handles both with and without trailing slash)
    path('parent/tasks/<uuid:pk>/delete/', 
         views.ParentTaskViewSet.as_view({'put': 'delete_task', 'delete': 'delete_task'}), 
         name='parent-task-delete'),
    path('parent/tasks/<uuid:pk>/delete',  # Without trailing slash
         views.ParentTaskViewSet.as_view({'put': 'delete_task', 'delete': 'delete_task'}), 
         name='parent-task-delete-noslash'),
    
    # Include router URLs
    path('', include(router.urls)),

    # for testing
    path('auth/google/callback/test', lambda request: HttpResponse("URL routing works!"), name='test'),
]

urlpatterns = [
    # API endpoints
    *api_patterns,
    
    # Legacy endpoints (consider deprecating these in the future)
    path('', views.index, name="index"),
    path('db_status/', views.db_status, name='db_status'),
    path('build_db/', views.build_db, name='build_db'),
]
