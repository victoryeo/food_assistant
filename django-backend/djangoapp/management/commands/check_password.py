from django.core.management.base import BaseCommand
import pymongo
from django.conf import settings
from passlib.context import CryptContext

class Command(BaseCommand):
    help = 'Check password for a user in MongoDB'

    def add_arguments(self, parser):
        parser.add_argument('email', type=str, help='Email of the user')
        parser.add_argument('password', type=str, help='Password to check')

    def handle(self, *args, **options):
        email = options['email']
        password = options['password']
        
        # Connect to MongoDB
        client = pymongo.MongoClient(settings.MONGODB_URI)
        db = client[settings.MONGODB_DB_NAME]
        
        # Find the user
        user = db.users.find_one({"email": email})
        if not user:
            self.stdout.write(self.style.ERROR(f'User with email {email} not found'))
            return
        
        hashed_password = user.get('hashed_password')
        if not hashed_password:
            self.stdout.write(self.style.ERROR('User has no password set'))
            return
        
        # Check the password
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        is_valid = pwd_context.verify(password, hashed_password)
        
        if is_valid:
            self.stdout.write(self.style.SUCCESS('Password is valid'))
        else:
            self.stdout.write(self.style.ERROR('Password is invalid'))
        
        self.stdout.write(f'Stored hash: {hashed_password}')
        client.close()
