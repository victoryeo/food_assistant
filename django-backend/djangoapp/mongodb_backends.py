import pymongo
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend
from django.contrib.auth.hashers import check_password
from passlib.context import CryptContext

class MongoDBAuthenticationBackend(ModelBackend):
    """
    Authenticate against MongoDB user documents.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize MongoDB client
        self.mongo_client = pymongo.MongoClient(settings.MONGODB_URI)
        self.db = self.mongo_client[settings.MONGODB_DB_NAME]
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        # Initialize _db to None, it will be set by Django
        self._db = None
    
    def authenticate(self, request, username=None, password=None, **kwargs):
        print(f"Username/Email: {username}")
        print(f"Password provided: {'Yes' if password else 'No'}")
        
        # Use email as the username
        email = username
        
        # Get user from MongoDB (synchronous)
        try:
            print(f"Looking up user with email: {email}")
            user_data = self.db.users.find_one({"email": email})
            
            if not user_data:
                print("User not found in database")
                return None
                
            print(f"Found user: {user_data}")
            print(f"User disabled status: {user_data.get('disabled', False)}")
            
            # Check password
            password_valid = self._check_password(password, user_data.get('hashed_password'))
            print(f"Password valid: {password_valid}")
            
            if not password_valid:
                print("Password validation failed")
                return None
                
            # If we get here, authentication was successful
            print("Authentication successful, getting or creating Django user...")
            user = self._get_or_create_user(email, user_data)
            print(f"Final user object: {user}")
            return user
                
        except Exception as e:
            print(f"Authentication error: {e}")
            import traceback
            traceback.print_exc()
            return None
    def _get_or_create_user(self, email, user_data):
        """Get or create a Django user based on MongoDB user data."""
        UserModel = get_user_model()
        
        try:
            # Try to get the user by email
            user = UserModel._default_manager.get(email=email)
            print(f"Found existing user: {user}")
        except UserModel.DoesNotExist:
            # Create a new user if one doesn't exist
            print(f"Creating new user with email: {email}")
            user = UserModel(email=email)
            # Save without using self._db to avoid the error
            user.save()
        
        # Update user data
        user.name = user_data.get('name', '')
        user.is_active = not user_data.get('disabled', False)
        
        # Set a dummy password to prevent password validation
        if not hasattr(user, 'password') or not user.password:
            user.set_unusable_password()
        
        # Save without using self._db to avoid the error
        user.save()
        print(f"Saved user: {user}")
        
        return user if self.user_can_authenticate(user) else None
    
    def _check_password(self, raw_password, hashed_password):
        """
        Check the password against the hashed password using passlib.
        """
        if not raw_password:
            print("No password provided")
            return False
            
        if not hashed_password:
            print("No hashed password found for user")
            return False
            
        try:
            print(f"Provided password: {raw_password}")
            print(f"Stored hash: {hashed_password}")
            
            # Verify the password
            is_valid = self.pwd_context.verify(raw_password, hashed_password)
            print(f"Password verification result: {is_valid}")
            return is_valid
            
        except Exception as e:
            print(f"Error verifying password: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_user(self, user_id):
        UserModel = get_user_model()
        try:
            user = UserModel._default_manager.get(pk=user_id)
            return user if self.user_can_authenticate(user) else None
        except UserModel.DoesNotExist:
            return None
