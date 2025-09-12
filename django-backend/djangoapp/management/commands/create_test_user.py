from django.core.management.base import BaseCommand
from django.conf import settings
from passlib.context import CryptContext
import asyncio

# Set up password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class Command(BaseCommand):
    help = 'Create a test user in MongoDB'

    async def create_test_user_async(self, email, password, name):
        """Create a test user in MongoDB asynchronously."""
        from motor.motor_asyncio import AsyncIOMotorClient
        import certifi
        
        # MongoDB connection
        client = AsyncIOMotorClient(
            settings.MONGODB_URI,
            tls=True,
            tlsCAFile=certifi.where(),
        )
        db = client[settings.MONGODB_DB_NAME]
        
        # Check if user already exists
        existing_user = await db.users.find_one({"email": email})
        if existing_user:
            self.stdout.write(self.style.WARNING(f'User {email} already exists'))
            return False
        
        # Create new user
        user_data = {
            "email": email,
            "hashed_password": pwd_context.hash(password),
            "name": name,
            "disabled": False
        }
        
        result = await db.users.insert_one(user_data)
        if result.inserted_id:
            self.stdout.write(
                self.style.SUCCESS(f'Successfully created test user {email} with ID: {result.inserted_id}')
            )
            return True
        return False

    def handle(self, *args, **options):
        email = "test@test.com"
        password = "test123"  # In a real app, you'd want to generate a secure password
        name = "Test User"
        
        # Run the async function
        loop = asyncio.get_event_loop()
        success = loop.run_until_complete(self.create_test_user_async(email, password, name))
        
        if not success:
            self.stdout.write(self.style.ERROR('Failed to create test user'))
