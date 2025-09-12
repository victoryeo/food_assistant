from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .models import Task, User, Student, Parent, TaskStatusHistory
from django.core.exceptions import ValidationError

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'email', 'name', 'picture', 'is_active', 'created_at']
        read_only_fields = ['is_active', 'created_at']
        extra_kwargs = {'password': {'write_only': True, 'required': False}}

    def validate_email(self, value):
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        return value

class StudentSerializer(serializers.ModelSerializer):
    user = UserSerializer(required=True)
    
    class Meta:
        model = Student
        fields = ['id', 'user', 'grade_level', 'date_of_birth', 'created_at']
        read_only_fields = ['created_at']

    def create(self, validated_data):
        user_data = validated_data.pop('user')
        user = User.objects.create_user(**user_data, is_student=True)
        student = Student.objects.create(user=user, **validated_data)
        return student

class ParentSerializer(serializers.ModelSerializer):
    user = UserSerializer(required=True)
    children = serializers.PrimaryKeyRelatedField(
        many=True,
        queryset=Student.objects.all(),
        required=False
    )
    
    class Meta:
        model = Parent
        fields = ['id', 'user', 'children', 'created_at']
        read_only_fields = ['created_at']

    def create(self, validated_data):
        user_data = validated_data.pop('user')
        children = validated_data.pop('children', [])
        user = User.objects.create_user(**user_data, is_parent=True)
        parent = Parent.objects.create(user=user, **validated_data)
        parent.children.set(children)
        return parent

class TaskStatusHistorySerializer(serializers.ModelSerializer):
    changed_by = UserSerializer(read_only=True)
    
    class Meta:
        model = TaskStatusHistory
        fields = ['id', 'status', 'changed_by', 'changed_at', 'notes']
        read_only_fields = ['changed_at']

class TaskSerializer(serializers.ModelSerializer):
    created_by = UserSerializer(read_only=True)
    assigned_to = UserSerializer()
    parent_task = serializers.PrimaryKeyRelatedField(queryset=Task.objects.all(), required=False, allow_null=True)
    status_history = TaskStatusHistorySerializer(many=True, read_only=True)
    
    class Meta:
        model = Task
        fields = [
            'id', 'title', 'description', 'task_type', 'completed',
            'due_date', 'created_at', 'updated_at', 'created_by',
            'assigned_to', 'parent_task', 'status_history'
        ]
        read_only_fields = ['created_at', 'updated_at', 'created_by']

    def validate_description(self, value):
        if not value or value.isspace():
            raise serializers.ValidationError('Message cannot be empty or contain only whitespace')
        return value

class TaskCreateSerializer(serializers.ModelSerializer):
    assigned_to = serializers.EmailField(write_only=True)
    
    class Meta:
        model = Task
        fields = ['title', 'description', 'task_type', 'due_date', 'assigned_to', 'parent_task']
        extra_kwargs = {
            'parent_task': {'required': False, 'allow_null': True}
        }
    
    def create(self, validated_data):
        assigned_to_email = validated_data.pop('assigned_to')
        try:
            assigned_to = User.objects.get(email=assigned_to_email)
        except User.DoesNotExist:
            raise serializers.ValidationError({"assigned_to": "User with this email does not exist."})
            
        task = Task.objects.create(
            assigned_to=assigned_to,
            created_by=self.context['request'].user,
            **validated_data
        )
        return task

class TaskUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Task
        fields = ['title', 'description', 'completed', 'due_date']
        extra_kwargs = {
            'title': {'required': False},
            'description': {'required': False},
            'due_date': {'required': False}
        }

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    def validate(self, attrs):
        # Get the email and password from the request
        email = attrs.get('email') or attrs.get('username')
        password = attrs.get('password')
        
        if not email or not password:
            raise serializers.ValidationError('Must include "email" and "password".')
        
        # Import here to avoid circular imports
        from django.contrib.auth import authenticate
        
        # Authenticate using our custom backend
        user = authenticate(
            request=self.context.get('request'),
            username=email,
            password=password
        )
        
        if not user:
            raise serializers.ValidationError('Unable to log in with provided credentials.')
        
        # If authentication is successful, generate tokens
        refresh = self.get_token(user)
        data = {
            'token_type': 'bearer',
            'refresh': str(refresh),
            'access_token': str(refresh.access_token),
            'user': {
                'id': str(user.id) if hasattr(user, 'id') else None,
                'email': user.email,
                'name': getattr(user, 'name', '')
            }
        }
        
        return data

class TaskSummarySerializer(serializers.Serializer):
    total_tasks = serializers.IntegerField()
    completed_tasks = serializers.IntegerField()
    pending_tasks = serializers.IntegerField()
    completion_rate = serializers.FloatField()
    tasks_by_status = serializers.DictField(child=serializers.IntegerField())
    recent_tasks = TaskSerializer(many=True)
    
    class Meta:
        fields = [
            'total_tasks', 'completed_tasks', 'pending_tasks',
            'completion_rate', 'tasks_by_status', 'recent_tasks'
        ]