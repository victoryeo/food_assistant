// src/app/api/users/route.ts
import fs from 'fs';
import path from 'path';
import { NextRequest } from 'next/server';
import { UserIntf } from '@/types/user';

const usersFile = path.join(process.cwd(), 'data', 'users.json');

const getUsersFromFile = () => {
  try {
    const data = fs.readFileSync(usersFile, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    return [];
  }
};

const saveUsersToFile = (users: any[]) => {
  fs.mkdirSync(path.dirname(usersFile), { recursive: true });
  fs.writeFileSync(usersFile, JSON.stringify(users, null, 2));
};

export async function POST(request: NextRequest) {
  const newUser = await request.json();
  console.log(newUser)
  const users = getUsersFromFile();
  
  if (users.find((u: any) => u.email === newUser.email)) {
    console.log("User already exists")
    return Response.json({ message: 'User already exists' }, { status: 400 });
  }
  
  const userToAdd: UserIntf = {
    ...newUser,
    id: Date.now().toString()
  };
  console.log(userToAdd)

  users.push(userToAdd);
  saveUsersToFile(users);
  console.log("userToAdd saved")
  return Response.json(userToAdd);
}

export async function GET(request: NextRequest) {
  try {
    // Get all users
    const data = fs.readFileSync(usersFile, 'utf8');
    return Response.json(JSON.parse(data));
  } catch (error) {
    return Response.json({ message: 'Internal server error' }, { status: 500 });
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const { email } = await request.json();
    
    if (!email) {
      return Response.json({ error: 'User email is required' }, { status: 400 });
    }
    
    const users = getUsersFromFile();
    const userIndex = users.findIndex((user: UserIntf) => user.email === email);
    
    if (userIndex === -1) {
      return Response.json({ error: 'User not found' }, { status: 404 });
    }
    
    // Remove the user from the array
    const deletedUser = users.splice(userIndex, 1)[0];
    
    // Save the updated users array back to the file
    saveUsersToFile(users);
    
    return Response.json({ 
      message: 'User deleted successfully',
      email: email 
    });
    
  } catch (error) {
    console.error('Error deleting user:', error);
    return Response.json(
      { error: 'Failed to delete user' }, 
      { status: 500 }
    );
  }
}