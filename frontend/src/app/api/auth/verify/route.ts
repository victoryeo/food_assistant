// src/app/api/auth/verify/route.ts
import fs from 'fs';
import path from 'path';
import { NextRequest } from 'next/server';
import { UserIntf } from '@/types/user';

export async function POST(request: NextRequest) {
  const { email, password } = await request.json();
  
  try {
    const usersFile = path.join(process.cwd(), 'data', 'users.json');
    const data = fs.readFileSync(usersFile, 'utf8');
    const users = JSON.parse(data);
    console.log("verify route")
    console.log(users)
    const user: UserIntf | undefined = users.find((u: UserIntf) => u.email === email && u.password === password);
    
    if (user) {
      return Response.json({
        id: user.id,
        email: user.email,
        name: user.name,
      });
    }
    
    return Response.json({ error: 'Invalid credentials' }, { status: 401 });
  } catch (error) {
    return Response.json({ error: 'Server error' }, { status: 500 });
  }
}