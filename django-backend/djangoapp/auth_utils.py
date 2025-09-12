import os
from datetime import timedelta
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils import timezone
from .mongodb_utils import get_user_by_email, create_user, update_user

User = get_user_model()

async def get_or_create_user_mongodb(google_user_data):
    """Get or create a user from Google OAuth data (async version for MongoDB)."""
    email = google_user_data.get('email')
    if not email:
        return None
    
    user = await get_user_by_email(email)
    if user:
        # Update user data if needed
        updates = {}
        if 'name' in google_user_data and google_user_data['name'] != user.get('name'):
            updates['name'] = google_user_data['name']
        if 'picture' in google_user_data and google_user_data['picture'] != user.get('picture'):
            updates['picture'] = google_user_data['picture']
        
        if updates:
            user = await update_user(user['id'], updates)
        return user
    else:
        # Create a new user
        user_data = {
            'email': email,
            'name': google_user_data.get('name', ''),
            'picture': google_user_data.get('picture', ''),
            'disabled': False
        }
        return await create_user(user_data)

# Keep the original function for backward compatibility
def get_or_create_user(google_user_data):
    """Synchronous wrapper for get_or_create_user_async."""
    import asyncio
    return asyncio.run(get_or_create_user_async(google_user_data))

def verify_google_token(token):
    """Verify Google OAuth token and return user data."""
    try:
        idinfo = id_token.verify_oauth2_token(
            token, 
            google_requests.Request(),
            settings.GOOGLE_CLIENT_ID
        )
        
        # Check if the token is valid
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            return None
            
        return idinfo
    except (ValueError, KeyError):
        return None

def get_tokens_for_user(user):
    """Generate JWT tokens for the user.
    
    Args:
        user: Can be either a Django User model instance or a MongoDB user document
    """
    from rest_framework_simplejwt.tokens import RefreshToken
    
    # Handle both Django User model and MongoDB user document
    if hasattr(user, 'id'):
        # Django User model
        user_id = str(user.id)
        email = user.email
        refresh = RefreshToken.for_user(user)
    else:
        # MongoDB user document
        user_id = str(user['id'])
        email = user['email']
        refresh = RefreshToken()
    
    # Set custom claims
    refresh['user_id'] = user_id
    refresh['email'] = email
    
    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
        'access_token_expiration': (timezone.now() + settings.SIMPLE_JWT['ACCESS_TOKEN_LIFETIME']).isoformat(),
        'refresh_token_expiration': (timezone.now() + settings.SIMPLE_JWT['REFRESH_TOKEN_LIFETIME']).isoformat(),
    }
