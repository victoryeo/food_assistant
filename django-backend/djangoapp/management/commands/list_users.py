from django.core.management.base import BaseCommand
import pymongo
from django.conf import settings
from pprint import pprint

class Command(BaseCommand):
    help = 'List all users in the MongoDB database'

    def handle(self, *args, **options):
        # Connect to MongoDB
        client = pymongo.MongoClient(settings.MONGODB_URI)
        db = client[settings.MONGODB_DB_NAME]
        
        # Get all users
        users = list(db.users.find({}))
        
        if not users:
            self.stdout.write(self.style.WARNING('No users found in the database'))
            return
            
        self.stdout.write(self.style.SUCCESS(f'Found {len(users)} users in the database:'))
        
        for user in users:
            # Convert ObjectId to string for display
            user_id = str(user.pop('_id'))
            self.stdout.write(f"\nUser ID: {user_id}")
            for key, value in user.items():
                # Don't print the hashed password for security
                if key == 'hashed_password':
                    self.stdout.write(f"{key}: {'*' * 8}")
                else:
                    self.stdout.write(f"{key}: {value}")
        
        client.close()
