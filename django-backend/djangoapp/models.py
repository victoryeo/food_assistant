from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.utils import timezone
import uuid
import re
from django.core.exceptions import ValidationError

class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        if password:
            user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(email, password, **extra_fields)

class User(AbstractBaseUser, PermissionsMixin):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(unique=True)
    name = models.CharField(max_length=100, blank=True, null=True)
    picture = models.URLField(blank=True, null=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    def __str__(self):
        return self.email

class Parent(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='parent_profile'
    )
    children = models.ManyToManyField('Student', related_name='parents')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Parent: {self.user.email}"

class Student(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='student_profile'
    )
    grade_level = models.CharField(max_length=20, blank=True, null=True)
    date_of_birth = models.DateField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Student: {self.user.email}"

def validate_task_message(value):
    """Validate task message to ensure it meets requirements."""
    if not value or value.isspace():
        raise ValidationError('Message cannot be empty or contain only whitespace')
    
    # Remove excessive whitespace
    value = ' '.join(value.split())
    
    # Check for minimum meaningful content
    if len(value.strip()) < 2:
        raise ValidationError('Message must contain at least 2 characters')

    # Check for profanity in various forms (case insensitive and with common substitutions)
    profanity_patterns = [
        r'f[\*u\$@]c[\*kq]',  # Matches f**k, f*ck, f*ck, fu*k, f**k, f*ck, etc.
        r'p[o0]rn',             # Matches porn, p0rn
        r'p[o0]rn[o0]',         # Matches porno, p0rn0
        r'wh[o0]re',            # Matches whore, wh0re
        r'sl[u\*]t',            # Matches slut, sl*t
        r'p[o0]rn[o0]gr[a@]phy', # Matches pornography, p0rn0graphy
    ]
    
    # Check if any profanity pattern matches
    for pattern in profanity_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            raise ValidationError('Message contains profanity')
    
    return value

class Task(models.Model):
    TASK_TYPES = [
        ('student', 'Student Task'),
        ('parent', 'Parent Task'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=200)
    description = models.TextField(validators=[validate_task_message])
    task_type = models.CharField(max_length=10, choices=TASK_TYPES)
    completed = models.BooleanField(default=False)
    due_date = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Relationships
    created_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='created_tasks'
    )
    assigned_to = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='assigned_tasks'
    )
    parent_task = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='subtasks'
    )

    def __str__(self):
        return f"{self.title} ({self.get_task_type_display()})"

    class Meta:
        ordering = ['-created_at']

class TaskStatusHistory(models.Model):
    task = models.ForeignKey(
        Task,
        on_delete=models.CASCADE,
        related_name='status_history'
    )
    status = models.BooleanField()  # True for completed, False for not completed
    changed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='task_status_changes'
    )
    changed_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True, null=True)

    class Meta:
        verbose_name_plural = 'Task status history'
        ordering = ['-changed_at']

    def __str__(self):
        status = 'Completed' if self.status else 'Incomplete'
        return f"{self.task.title} - {status} at {self.changed_at}"