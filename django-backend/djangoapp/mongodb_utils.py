import os
import certifi
from motor.motor_asyncio import AsyncIOMotorClient
from django.conf import settings
from bson import ObjectId
from datetime import datetime

# MongoDB connection setup
MONGODB_URI = settings.MONGODB_URI
DB_NAME = settings.MONGODB_DB_NAME

# For async operations
async_client = AsyncIOMotorClient(
    MONGODB_URI,
    tls=True,
    tlsCAFile=certifi.where(),
)
db = async_client[DB_NAME]

# Password hashing
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def get_user_by_email(email: str):
    """Get user by email from MongoDB"""
    try:
        user_data = await db.users.find_one({"email": email})
        if user_data:
            # Convert ObjectId to string for JSON serialization
            user_data["id"] = str(user_data["_id"])
            return user_data
        return None
    except Exception as e:
        print(f"Database error: {e}")
        return None

async def create_user(user_data: dict):
    """Create new user in MongoDB"""
    try:
        # Hash password if provided
        if 'password' in user_data and user_data['password']:
            user_data['hashed_password'] = pwd_context.hash(user_data.pop('password'))
        
        # Set timestamps
        now = datetime.utcnow()
        user_data['created_at'] = now
        user_data['updated_at'] = now
        
        # Insert user into database
        result = await db.users.insert_one(user_data)
        if result.inserted_id:
            user_data['id'] = str(result.inserted_id)
            return user_data
        return None
    except Exception as e:
        print(f"Database error: {e}")
        return None

async def update_user(user_id: str, update_data: dict):
    """Update user in MongoDB"""
    try:
        update_data['updated_at'] = datetime.utcnow()
        result = await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        if result.modified_count == 1:
            return await get_user_by_id(user_id)
        return None
    except Exception as e:
        print(f"Database error: {e}")
        return None

async def get_user_by_id(user_id: str):
    """Get user by ID from MongoDB"""
    try:
        user_data = await db.users.find_one({"_id": ObjectId(user_id)})
        if user_data:
            user_data["id"] = str(user_data["_id"])
            return user_data
        return None
    except Exception as e:
        print(f"Database error: {e}")
        return None
